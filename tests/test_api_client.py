import json
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from harpq import api_client as api


def configured(candidate="M04"):
    return replace(api.MODEL_CONFIGS[candidate], api_key="offline-test",
                   base_url="https://example.invalid/v1")


class StructuredResponseTests(unittest.TestCase):
    def test_blank_credentials_and_endpoint_prevent_requests(self):
        with patch.object(api, "_complete") as complete:
            for candidate in api.MODEL_CONFIGS.values():
                for config in (candidate, replace(candidate, api_key="offline-test")):
                    with self.assertRaises(ValueError):
                        api.request_json(config, "System", "Task")
            complete.assert_not_called()
        self.assertNotIn("offline-test", repr(configured()))

    def test_retry_preserves_model_and_task(self):
        with patch.object(api, "_complete", side_effect=["not JSON", '{"value": 2}']) as complete:
            result = api.request_json(configured(), "System", "Task")
        self.assertEqual(result, api.APIResult({"value": 2}, 2))
        first, second = complete.call_args_list
        self.assertEqual(first.args[:2], second.args[:2])
        self.assertTrue(second.args[2].startswith(first.args[2]))

    def test_invalid_json_is_never_silently_accepted(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}',
                     '{"x":1e400}', '[1,2]', '```json\n{}\n```', ''):
            with self.subTest(text=text), patch.object(api, "_complete", return_value=text) as complete:
                with self.assertRaises(api.StructuredResponseError) as caught:
                    api.request_json(configured(), "System", "Task")
                self.assertEqual(caught.exception.attempts, 2)
                self.assertEqual(complete.call_count, 2)

    def test_two_nodes_share_one_request_and_continuous_clipping(self):
        nodes = {"wholesaler": api.NodeState((50,), 100, 0, 0),
                 "distributor": api.NodeState((50,), 100, 0, 0)}
        response = '{"orders":{"wholesaler":12.75,"distributor":220}}'
        with patch.object(api, "_complete", return_value=response) as complete:
            result = api.request_bg_orders(configured(), "System", "Task", nodes)
        self.assertEqual(result.payload["orders"], {"wholesaler":12.75, "distributor":200.0})
        self.assertEqual(result.attempts, 1)
        self.assertFalse(result.used_fallback)
        complete.assert_called_once()

    def test_second_failure_applies_node_specific_fallback(self):
        nodes = {"wholesaler": api.NodeState((40, 60), 180, 40, 10),
                 "distributor": api.NodeState((10, 20, 30, 40, 50), 400, 0, 0)}
        with patch.object(api, "_complete", return_value='{"orders":{"wholesaler":true}}') as complete:
            result = api.request_bg_orders(configured(), "System", "Task", nodes)
        self.assertEqual(result, api.APIResult({"orders":{"wholesaler":90.0, "distributor":0.0}}, 2, True))
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(api.fallback_order(api.NodeState((), 50, 0, 0)), 200.0)
        self.assertEqual(api.fallback_order(api.NodeState((40, 40, 40, 40), 200, 0, 0)), 60.0)

    def test_unknown_nodes_and_boolean_orders_are_rejected(self):
        nodes = {"retailer": api.NodeState((), 300, 0, 0)}
        for orders in ({"retailer": True}, {"retailer": 50, "factory": 50}, {}, {"retailer": "50"}):
            with self.subTest(orders=orders), patch.object(api, "_complete", return_value=json.dumps({"orders":orders})):
                self.assertTrue(api.request_bg_orders(configured(), "System", "Task", nodes).used_fallback)

    def test_transport_error_is_not_a_format_retry_or_fallback(self):
        with patch.object(api, "_complete", side_effect=ConnectionError("offline")) as complete:
            with self.assertRaises(ConnectionError):
                api.request_bg_orders(configured(), "System", "Task",
                                      {"retailer": api.NodeState((), 100, 0, 0)})
        complete.assert_called_once()

    def test_invalid_state_and_empty_prompt_fail_before_call(self):
        with patch.object(api, "_complete") as complete:
            for state in (api.NodeState((-1,), 0, 0, 0), api.NodeState((), float("nan"), 0, 0)):
                with self.assertRaises(ValueError):
                    api.request_bg_orders(configured(), "System", "Task", {"retailer":state})
            with self.assertRaises(ValueError):
                api.request_bg_orders(configured(), "System", "", {"retailer":api.NodeState((), 0, 0, 0)})
            complete.assert_not_called()


class ProviderAdapterTests(unittest.TestCase):
    def test_openai_compatible_payloads_and_retry_settings(self):
        for candidate in ("M01", "M02", "M03", "M04"):
            client = MagicMock()
            client.__enter__.return_value = client
            client.chat.completions.create.return_value = SimpleNamespace(choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"ok":true}', refusal=None))
            ])
            factory = MagicMock(return_value=client)
            with self.subTest(candidate=candidate), patch.dict("sys.modules", {"openai":SimpleNamespace(OpenAI=factory)}):
                self.assertEqual(api.request_json(configured(candidate), "System", "Task").payload, {"ok":True})
            self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
            parameters = client.chat.completions.create.call_args.kwargs
            self.assertEqual(parameters["model"], api.MODEL_CONFIGS[candidate].model)
            self.assertEqual(parameters["temperature"], .20)
            self.assertEqual(parameters["top_p"], .95)
            token_key = "max_completion_tokens" if candidate in {"M03", "M04"} else "max_tokens"
            self.assertEqual(parameters[token_key], 1024)
            self.assertEqual(parameters["response_format"], {"type":"json_object"})
            if candidate == "M02":
                self.assertEqual(parameters["extra_body"], {"thinking":{"type":"disabled"}})
            client.__exit__.assert_called_once()

    def test_google_payload_retry_settings_and_thought_exclusion(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.generate_content.return_value = SimpleNamespace(candidates=[
            SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[
                SimpleNamespace(text="hidden", thought=True),
                SimpleNamespace(text='{"ok":true}', thought=False),
            ]))
        ])
        factory = MagicMock(return_value=client)
        types = SimpleNamespace(HttpOptions=SimpleNamespace, HttpRetryOptions=SimpleNamespace,
                                GenerateContentConfig=SimpleNamespace)
        genai = SimpleNamespace(Client=factory, types=types)
        with patch.dict("sys.modules", {"google":SimpleNamespace(genai=genai), "google.genai":genai}):
            result = api.request_json(configured("M05"), "System", "Task")
        self.assertEqual(result.payload, {"ok":True})
        self.assertFalse(factory.call_args.kwargs["vertexai"])
        options = factory.call_args.kwargs["http_options"]
        self.assertEqual(options.retry_options.attempts, 1)
        self.assertEqual(options.timeout, 60000)
        parameters = client.models.generate_content.call_args.kwargs
        self.assertEqual(parameters["model"], "gemini-2.5-flash-lite")
        self.assertEqual(parameters["config"].max_output_tokens, 1024)
        self.assertEqual(parameters["config"].temperature, .20)
        self.assertEqual(parameters["config"].top_p, .95)
        client.__exit__.assert_called_once()

    def test_truncated_response_does_not_pass_even_if_json_is_valid(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"ok":true}', refusal=None))
        ])
        with patch.dict("sys.modules", {"openai":SimpleNamespace(OpenAI=MagicMock(return_value=client))}):
            with self.assertRaises(api.StructuredResponseError):
                api.request_json(configured(), "System", "Task")
        self.assertEqual(client.chat.completions.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
