import json
from pathlib import Path
import unittest
from unittest.mock import patch

from harpq import bg
from harpq.api_client import MODEL_CONFIGS, request_bg_orders
from dataclasses import replace


def chain(initial_message=0.0, priority="backlog_first"):
    return bg.BGChain(initial_order_messages={n:(initial_message, initial_message) for n in bg.NODES[1:]},
                      fulfillment_priority=priority)


def orders(value=0.0, **changes):
    return {**dict.fromkeys(bg.NODES, value), **changes}


def finish(environment):
    while environment.period < bg.PERIODS:
        environment.begin_period(0.0)
        environment.submit_orders(orders())


class BGTests(unittest.TestCase):
    def test_initial_inventory_and_each_pipeline(self):
        environment = chain()
        first = environment.begin_period(0.0)
        self.assertEqual([s.on_hand for s in first.values()], [100.0]*4)
        self.assertEqual([s.backlog for s in first.values()], [0.0]*4)
        self.assertEqual([s.outstanding_inbound for s in first.values()], [50.0]*4)
        self.assertEqual(environment.submit_orders(orders()).receipts, (50.0,)*4)
        second = environment.begin_period(0.0)
        self.assertEqual([s.on_hand for s in second.values()], [150.0]*4)
        self.assertEqual(environment.submit_orders(orders()).receipts, (50.0,)*4)
        environment.begin_period(0.0)
        self.assertEqual(environment.submit_orders(orders()).receipts, (0.0,)*4)

    def test_information_shipping_and_production_delays(self):
        environment = chain()
        environment.begin_period(0.0)
        environment.submit_orders(orders(retailer=7.5, factory=9.25))
        environment.begin_period(0.0)
        self.assertEqual(environment.submit_orders(orders()).demand, (0.0,)*4)
        states = environment.begin_period(0.0)
        third = environment.submit_orders(orders())
        self.assertEqual(third.demand, (0.0,7.5,0.0,0.0))
        self.assertEqual(third.receipts, (0.0,0.0,0.0,9.25))
        self.assertEqual(third.shipments[1], 7.5)
        self.assertEqual(states["retailer"].outstanding_inbound, 7.5)
        environment.begin_period(0.0)
        self.assertEqual(environment.submit_orders(orders()).receipts[0], 0.0)
        environment.begin_period(0.0)
        self.assertEqual(environment.submit_orders(orders()).receipts[0], 7.5)

    def test_backlog_clearance_does_not_improve_immediate_service(self):
        environment = chain()
        environment.begin_period(150.0)
        first = environment.submit_orders(orders(retailer=50.0))
        self.assertEqual(first.backlog[0], 50.0)
        environment.begin_period(50.0)
        second = environment.submit_orders(orders())
        self.assertEqual(second.shipments[0], 50.0)
        self.assertEqual(second.immediately_fulfilled[0], 0.0)
        finish(environment)
        self.assertEqual(environment.terminal_state()["nodes"]["retailer"]["backlog"], 0.0)
        self.assertEqual(sum(r.shipments[0] for r in environment.records), 200.0)
        self.assertEqual(environment.outcomes()["service"], 0.5)
        self.assertEqual(environment.records[4].immediately_fulfilled[0], 0.0)

    def test_priority_is_an_explicit_run_input(self):
        environment = chain(priority="current_first")
        environment.begin_period(150.0)
        environment.submit_orders(orders(retailer=50.0))
        environment.begin_period(50.0)
        self.assertEqual(environment.submit_orders(orders()).immediately_fulfilled[0], 50.0)
        finish(environment)
        self.assertEqual(environment.outcomes()["service"], .75)
        with self.assertRaises(TypeError):
            bg.BGChain()

    def test_end_of_horizon_retains_backlog_pipeline_and_messages(self):
        environment = chain()
        for _ in range(35):
            environment.begin_period(0.0)
            environment.submit_orders(orders())
        environment.begin_period(200.0)
        last = environment.submit_orders(orders(retailer=25.5, factory=70.75))
        terminal = environment.terminal_state()
        self.assertEqual(last.backlog[0], 50.0)
        self.assertEqual(terminal["nodes"]["retailer"]["backlog"], 50.0)
        self.assertEqual(terminal["nodes"]["factory"]["pipeline"], [{"due_period":38,"quantity":70.75}])
        self.assertEqual(terminal["nodes"]["wholesaler"]["order_messages"], [{"due_period":38,"quantity":25.5}])
        self.assertEqual(environment.outcomes()["service"], .75)
        with self.assertRaises(RuntimeError):
            environment.begin_period(0.0)
        self.assertEqual(environment.terminal_state(), terminal)

    def test_constant_flow_cost_is_charged_once_per_period_per_node(self):
        environment = chain(initial_message=50.0)
        for _ in range(36):
            environment.begin_period(50.0)
            record = environment.submit_orders(orders(50.0))
            self.assertEqual(record.on_hand, (50.0,)*4)
            self.assertEqual(record.backlog, (0.0,)*4)
            self.assertEqual(record.cost_per_node, 50.0)
        self.assertEqual(environment.outcomes(), {"cost":1800.0,"bullwhip":None,"service":1.0})
        self.assertEqual(len(environment.records), 36)

    def test_bullwhip_uses_all_36_executed_factory_plans(self):
        environment = chain()
        for t in range(1,37):
            environment.begin_period(float(t))
            environment.submit_orders(orders(factory=2*t))
        self.assertAlmostEqual(environment.outcomes()["bullwhip"], 4.0)

    def test_clipping_keeps_fractional_orders(self):
        environment = chain()
        environment.begin_period(0.0)
        record = environment.submit_orders(orders(retailer=-2.0, wholesaler=12.75, distributor=201.5))
        self.assertEqual(record.orders, (0.0,12.75,200.0,0.0))

    def test_order_decisions_and_outcomes_follow_period_phases(self):
        environment = chain()
        for action in (environment.decision_states, environment.outcomes, environment.terminal_state):
            with self.assertRaises(RuntimeError):
                action()
        with self.assertRaises(RuntimeError):
            environment.submit_orders(orders())
        environment.begin_period(0.0)
        with self.assertRaises(RuntimeError):
            environment.begin_period(0.0)
        with self.assertRaises(ValueError):
            environment.submit_orders(orders(factory=float("nan")))
        environment.submit_orders(orders())
        self.assertEqual(len(environment.records), 1)

    def test_receipts_and_shipments_conserve_quantities(self):
        environment = chain(initial_message=30.0)
        previous_stock = [50.0]*4
        previous_backlog = [0.0]*4
        for t in range(36):
            environment.begin_period(float(35+(13*t)%55))
            record = environment.submit_orders({n:float((31*t+17*i)%201) for i,n in enumerate(bg.NODES)})
            for i in range(4):
                self.assertAlmostEqual(record.on_hand[i], previous_stock[i]+record.receipts[i]-record.shipments[i])
                self.assertAlmostEqual(record.backlog[i], previous_backlog[i]+record.demand[i]-record.shipments[i])
                self.assertLessEqual(record.immediately_fulfilled[i], record.demand[i])
            previous_stock, previous_backlog = record.on_hand, record.backlog
        terminal = environment.terminal_state()
        final_stock = sum(v["on_hand"]+sum(p["quantity"] for p in v["pipeline"]) for v in terminal["nodes"].values())
        self.assertAlmostEqual(600.0+sum(r.orders[-1]-r.shipments[0] for r in environment.records), final_stock)
        self.assertAlmostEqual(sum(r.cost_per_node for r in environment.records), environment.outcomes()["cost"])

    def test_reset_restores_initial_state_without_carryover(self):
        environment = chain()
        environment.begin_period(80.0)
        first = environment.submit_orders(orders(20.25))
        finish(environment)
        environment.reset()
        self.assertEqual(environment.period, 0)
        self.assertEqual(environment.records, ())
        environment.begin_period(80.0)
        self.assertEqual(environment.submit_orders(orders(20.25)), first)

    def test_api_fallback_uses_post_fulfillment_node_states(self):
        environment = chain()
        all_states = environment.begin_period(0.0)
        nodes = {node:all_states[node] for node in ("wholesaler","distributor")}
        config = replace(MODEL_CONFIGS["M02"], api_key="offline-test", base_url="https://example.invalid/v1")
        with patch("harpq.api_client._complete", return_value="invalid") as complete:
            result = request_bg_orders(config, "System", "Task", nodes)
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(result.payload, {"orders":{"wholesaler":100.0,"distributor":100.0}})
        environment.submit_orders({**orders(), **result.payload["orders"]})

    def test_protocol_json_matches_fixed_settings(self):
        path = Path(__file__).resolve().parents[1]/"data"/"bg_protocol.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["nodes"], list(bg.NODES))
        self.assertEqual(protocol["periods"], bg.PERIODS)
        self.assertEqual(protocol["initial_on_hand_per_node"], bg.INITIAL_ON_HAND)
        self.assertEqual(protocol["initial_backlog_per_node"], bg.INITIAL_BACKLOG)
        self.assertEqual(tuple(r["quantity"] for r in protocol["initial_receipts_per_pipeline"]), bg.INITIAL_RECEIPTS)
        self.assertEqual([r["due_period"] for r in protocol["initial_receipts_per_pipeline"]], [1,2])
        for name in ("order_information_delay","shipping_lead_time","production_lead_time"):
            self.assertEqual(protocol[name], getattr(bg,name.upper()))
        self.assertEqual(protocol["cost"], {"holding_rate":bg.HOLDING_RATE,"backlog_rate":bg.BACKLOG_RATE,"normalization_nodes":4})
        self.assertEqual(protocol["actions"], {"lower":bg.ACTION_BOUNDS[0],"upper":bg.ACTION_BOUNDS[1],"integer_rounding":False})


if __name__ == "__main__":
    unittest.main()
