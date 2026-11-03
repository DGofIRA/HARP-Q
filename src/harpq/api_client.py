"""Provider API adapters and structured-response handling for HARP-Q."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit


TEMPERATURE = 0.20
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 1024


@dataclass(frozen=True)
class APIConfig:
    provider: str
    model: str
    api_key: str = field(default="", repr=False)
    base_url: str = field(default="", repr=False)
    timeout_seconds: float = 60.0

    def validate(self) -> None:
        if self.provider not in {"openai", "qwen", "deepseek", "google"}:
            raise ValueError("Unsupported API provider.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("A model identifier is required.")
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("API credentials are not configured.")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("The API endpoint is not configured.")
        endpoint = urlsplit(self.base_url)
        if (endpoint.scheme != "https" or not endpoint.hostname
                or endpoint.username or endpoint.password
                or endpoint.query or endpoint.fragment):
            raise ValueError("The API endpoint must be an HTTPS base URL.")
        if _finite_number(self.timeout_seconds) <= 0:
            raise ValueError("The request timeout must be positive.")


MODEL_CONFIGS = MappingProxyType({
    "M01": APIConfig("qwen", "qwen2.5-72b-instruct"),
    "M02": APIConfig("deepseek", "deepseek-v4-pro"),
    "M03": APIConfig("openai", "gpt-4o-2024-11-20"),
    "M04": APIConfig("openai", "gpt-4.1-2025-04-14"),
    "M05": APIConfig("google", "gemini-2.5-flash-lite"),
})
PLACEMENT_SLOTS = MappingProxyType({"L01": "M05", "L02": "M02", "L03": "M04"})


@dataclass(frozen=True)
class APIResult:
    payload: dict[str, Any]
    attempts: int
    used_fallback: bool = False


@dataclass(frozen=True)
class NodeState:
    demand_history: tuple[float, ...]
    on_hand: float
    outstanding_inbound: float
    backlog: float


class StructuredResponseError(ValueError):
    def __init__(self) -> None:
        super().__init__("No valid structured response after two attempts.")
        self.attempts = 2


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A finite numeric value is required.")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("A finite numeric value is required.") from None
    if not math.isfinite(number):
        raise ValueError("A finite numeric value is required.")
    return number


def _json_object(text: str) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result

    def reject_constant(_):
        raise ValueError("Non-finite JSON number.")

    def finite_float(value):
        return _finite_number(float(value))

    result = json.loads(text, object_pairs_hook=unique_pairs,
                        parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(result, dict):
        raise ValueError("The response must be a JSON object.")
    return result


def _complete(config: APIConfig, system: str, prompt: str) -> str:
    if config.provider == "google":
        from google import genai
        from google.genai import types

        options = types.HttpOptions(
            base_url=config.base_url,
            timeout=max(1, int(config.timeout_seconds * 1000)),
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        with genai.Client(api_key=config.api_key, vertexai=False,
                          http_options=options) as client:
            response = client.models.generate_content(
                model=config.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    candidate_count=1,
                ),
            )
        if not response.candidates:
            return ""
        candidate = response.candidates[0]
        reason = getattr(candidate.finish_reason, "value", candidate.finish_reason)
        if reason != "STOP" or candidate.content is None:
            return ""
        return "".join(part.text for part in (candidate.content.parts or [])
                       if part.text and not part.thought)

    from openai import OpenAI

    parameters = {
        "model": config.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    token_key = "max_completion_tokens" if config.provider == "openai" else "max_tokens"
    parameters[token_key] = MAX_OUTPUT_TOKENS
    if config.provider == "deepseek":
        parameters["extra_body"] = {"thinking": {"type": "disabled"}}
    with OpenAI(api_key=config.api_key, base_url=config.base_url,
                timeout=config.timeout_seconds, max_retries=0) as client:
        response = client.chat.completions.create(**parameters)
    if not response.choices:
        return ""
    choice = response.choices[0]
    if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
        return ""
    return choice.message.content or ""


def request_json(
    config: APIConfig,
    system: str,
    prompt: str,
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> APIResult:
    """Return validated JSON; transport and authentication errors propagate."""
    config.validate()
    if not all(isinstance(value, str) and value.strip() for value in (system, prompt)):
        raise ValueError("Nonempty system instructions and a task prompt are required.")
    if validator is not None and not callable(validator):
        raise TypeError("The response validator must be callable.")
    system = system + "\nReturn one JSON object only, without Markdown fences."
    for attempt in (1, 2):
        request_prompt = prompt
        if attempt == 2:
            request_prompt += ("\nThe preceding response did not satisfy the required JSON "
                               "format or fields. Return the required JSON object with "
                               "all specified fields and valid values.")
        text = _complete(config, system, request_prompt)
        try:
            payload = _json_object(text)
            if validator is not None:
                payload = validator(payload)
            if not isinstance(payload, dict):
                raise ValueError("The response validator must return a JSON object.")
        except ValueError:
            continue
        return APIResult(payload, attempt)
    raise StructuredResponseError()


def fallback_order(state: NodeState) -> float:
    recent = [_finite_number(value) for value in state.demand_history[-4:]]
    stocks = [_finite_number(state.on_hand),
              _finite_number(state.outstanding_inbound), _finite_number(state.backlog)]
    if any(value < 0 for value in recent + stocks):
        raise ValueError("Demand, on-hand stock, inbound stock, and backlog must be nonnegative.")
    recent = [50.0] * (4 - len(recent)) + recent
    inventory_position = stocks[0] + stocks[1] - stocks[2]
    order = sum(recent) + 100.0 - inventory_position
    if not math.isfinite(order):
        raise ValueError("Inventory arithmetic exceeded the finite numeric range.")
    return max(0.0, min(200.0, order))


def request_bg_orders(
    config: APIConfig,
    system: str,
    prompt: str,
    nodes: Mapping[str, NodeState],
) -> APIResult:
    """Return one response for the controlled BG nodes, or their common fallback."""
    if not nodes or any(not isinstance(key, str) or not key.strip() for key in nodes):
        raise ValueError("At least one named node is required.")
    fallback = {name: fallback_order(state) for name, state in nodes.items()}
    names = tuple(nodes)

    def validate(payload):
        orders = payload.get("orders")
        if (set(payload) != {"orders"} or not isinstance(orders, dict)
                or set(orders) != set(names)):
            raise ValueError("The response must contain exactly the requested node orders.")
        return {"orders": {name: max(0.0, min(200.0, _finite_number(orders[name])))
                           for name in names}}

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty task prompt is required.")
    schema = ("\nReturn a JSON object with exactly one field, orders. Its value must be "
              "an object containing exactly these node keys: " + json.dumps(names)
              + ". Each order must be a finite numeric quantity in [0, 200].")
    try:
        return request_json(config, system, prompt + schema, validate)
    except StructuredResponseError as error:
        return APIResult({"orders": fallback}, error.attempts, used_fallback=True)
