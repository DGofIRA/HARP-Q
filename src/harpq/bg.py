"""Four-echelon BG state transitions and finite-horizon outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from .api_client import NodeState
from .metrics import fulfillment_statistics, normalized_chain_cost, variance_ratio


NODES = ("retailer", "wholesaler", "distributor", "factory")
PERIODS = 36
INITIAL_ON_HAND = 50.0
INITIAL_BACKLOG = 0.0
INITIAL_RECEIPTS = (50.0, 50.0)
ORDER_INFORMATION_DELAY = 2
SHIPPING_LEAD_TIME = 2
PRODUCTION_LEAD_TIME = 2
HOLDING_RATE = 1.0
BACKLOG_RATE = 2.0
ACTION_BOUNDS = (0.0, 200.0)


def _quantity(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A finite numeric quantity is required.")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError("A finite numeric quantity is required.") from None
    if not isfinite(value):
        raise ValueError("A finite numeric quantity is required.")
    return value


def _nonnegative(value):
    value = _quantity(value)
    if value < 0:
        raise ValueError("Demand and initial order messages must be nonnegative.")
    return value


@dataclass(frozen=True)
class PeriodRecord:
    period: int
    receipts: tuple[float, ...]
    demand: tuple[float, ...]
    shipments: tuple[float, ...]
    immediately_fulfilled: tuple[float, ...]
    on_hand: tuple[float, ...]
    backlog: tuple[float, ...]
    orders: tuple[float, ...]
    cost_per_node: float


class BGChain:
    """Initial order messages and fulfillment priority are explicit run inputs."""

    def __init__(self, *, initial_order_messages: Mapping[str, Sequence[float]],
                 fulfillment_priority: str):
        if set(initial_order_messages) != set(NODES[1:]):
            raise ValueError("Initial order messages must specify the three upstream nodes.")
        self._initial_messages = {}
        for node in NODES[1:]:
            messages = tuple(initial_order_messages[node])
            if len(messages) != ORDER_INFORMATION_DELAY:
                raise ValueError("Initial order messages require arrivals for periods 1 and 2.")
            self._initial_messages[node] = tuple(_nonnegative(x) for x in messages)
        if fulfillment_priority not in {"backlog_first", "current_first"}:
            raise ValueError("A valid fulfillment priority is required.")
        self._priority = fulfillment_priority
        self.reset()

    def reset(self) -> None:
        self._period = 0
        self._on_hand = [INITIAL_ON_HAND] * len(NODES)
        self._backlog = [INITIAL_BACKLOG] * len(NODES)
        self._receipts = [{period: quantity for period, quantity in enumerate(INITIAL_RECEIPTS, 1)}
                          for _ in NODES]
        self._messages = [{}] + [dict(enumerate(self._initial_messages[node], 1))
                                 for node in NODES[1:]]
        self._demands = [[] for _ in NODES]
        self._records = []
        self._pending = None

    @property
    def records(self) -> tuple[PeriodRecord, ...]:
        return tuple(self._records)

    @property
    def period(self) -> int:
        return self._period

    def begin_period(self, external_demand: float) -> dict[str, NodeState]:
        if self._pending is not None:
            raise RuntimeError("Orders for the current period have not been submitted.")
        if self._period >= PERIODS:
            raise RuntimeError("The 36-period horizon is complete.")
        external_demand = _nonnegative(external_demand)
        period = self._period + 1
        receipts = [pipeline.get(period, 0.0) for pipeline in self._receipts]
        demand = [external_demand] + [pipeline.get(period, 0.0) for pipeline in self._messages[1:]]
        on_hand, backlog, shipments, immediate = [], [], [], []
        for i in range(len(NODES)):
            available = self._on_hand[i] + receipts[i]
            required = self._backlog[i] + demand[i]
            if not isfinite(available) or not isfinite(required):
                raise ValueError("Inventory arithmetic exceeded the finite numeric range.")
            sent = min(available, required)
            current = min(demand[i], max(0.0, available - self._backlog[i])) \
                if self._priority == "backlog_first" else min(demand[i], available)
            shipments.append(sent)
            immediate.append(current)
            on_hand.append(available - sent)
            backlog.append(required - sent)
        for pipeline in self._receipts + self._messages:
            pipeline.pop(period, None)
        for i in range(1, len(NODES)):
            due = period + SHIPPING_LEAD_TIME
            if shipments[i]:
                self._receipts[i - 1][due] = self._receipts[i - 1].get(due, 0.0) + shipments[i]
        self._period = period
        self._on_hand, self._backlog = on_hand, backlog
        for history, quantity in zip(self._demands, demand):
            history.append(quantity)
        self._pending = {
            "receipts": tuple(receipts), "demand": tuple(demand),
            "shipments": tuple(shipments), "immediately_fulfilled": tuple(immediate),
        }
        return self.decision_states()

    def decision_states(self) -> dict[str, NodeState]:
        if self._pending is None:
            raise RuntimeError("Decision states are available after receipt and fulfillment processing.")
        states = {}
        for i, node in enumerate(NODES):
            outstanding = sum(self._receipts[i].values())
            if i + 1 < len(NODES):
                outstanding += sum(self._messages[i + 1].values()) + self._backlog[i + 1]
            states[node] = NodeState(tuple(self._demands[i]), self._on_hand[i],
                                     outstanding, self._backlog[i])
        return states

    def submit_orders(self, approved_orders: Mapping[str, float]) -> PeriodRecord:
        if self._pending is None:
            raise RuntimeError("A period must begin before orders are submitted.")
        if set(approved_orders) != set(NODES):
            raise ValueError("Approved orders must specify all four nodes exactly once.")
        lower, upper = ACTION_BOUNDS
        orders = tuple(max(lower, min(upper, _quantity(approved_orders[node]))) for node in NODES)
        cost = normalized_chain_cost(self._on_hand, self._backlog, len(NODES),
                                     HOLDING_RATE, BACKLOG_RATE)
        if not isfinite(cost):
            raise ValueError("Cost arithmetic exceeded the finite numeric range.")
        for i, order in enumerate(orders):
            if not order:
                continue
            if i + 1 < len(NODES):
                pipeline = self._messages[i + 1]
                due = self._period + ORDER_INFORMATION_DELAY
            else:
                pipeline = self._receipts[i]
                due = self._period + PRODUCTION_LEAD_TIME
            pipeline[due] = pipeline.get(due, 0.0) + order
        record = PeriodRecord(self._period, **self._pending, on_hand=tuple(self._on_hand),
                              backlog=tuple(self._backlog), orders=orders, cost_per_node=cost)
        self._records.append(record)
        self._pending = None
        return record

    def outcomes(self) -> dict[str, float | None]:
        self._require_complete()
        demand = [r.demand[0] for r in self._records]
        fulfilled = [r.immediately_fulfilled[0] for r in self._records]
        return {
            "cost": normalized_chain_cost(
                [x for r in self._records for x in r.on_hand],
                [x for r in self._records for x in r.backlog], len(NODES),
                HOLDING_RATE, BACKLOG_RATE),
            "bullwhip": variance_ratio([r.orders[-1] for r in self._records], demand),
            "service": fulfillment_statistics(demand, fulfilled)["service"],
        }

    def terminal_state(self) -> dict:
        self._require_complete()
        return {
            "period": self._period,
            "fulfillment_priority": self._priority,
            "nodes": {node: {
                "on_hand": self._on_hand[i], "backlog": self._backlog[i],
                "pipeline": [{"due_period": due, "quantity": amount}
                             for due, amount in sorted(self._receipts[i].items())],
                "order_messages": [{"due_period": due, "quantity": amount}
                                   for due, amount in sorted(self._messages[i].items())],
            } for i, node in enumerate(NODES)},
        }

    def _require_complete(self) -> None:
        if len(self._records) != PERIODS or self._pending is not None:
            raise RuntimeError("Outcomes require a complete 36-period trajectory.")
