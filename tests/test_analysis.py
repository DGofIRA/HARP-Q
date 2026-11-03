import json
from pathlib import Path
import unittest

from harpq.scoring import normalize, harmonic, score_panel, qualifications
from harpq.assignment import assignments, assignment_summary
from harpq.diagnostics import diagnostics
from harpq.metrics import planning_distance, variance_ratio, local_chasing, fulfillment_statistics, normalized_chain_cost
from harpq.statistics import holm, mean_interval, paired_cluster_summary

DATA = Path(__file__).resolve().parents[1] / "data"


def read(name):
    return json.loads((DATA / (name + ".json")).read_text(encoding="utf8"))


class ScoringTests(unittest.TestCase):
    def test_rank_ties_and_constant_columns(self):
        self.assertEqual(normalize([2, 2, 4, 1]), [.5, .5, .875, .125])
        for method in ("midpoint", "endpoint", "minmax", "compressed_minmax"):
            self.assertEqual(normalize([7, 7, 7], method), [.5, .5, .5])

    def test_invalid_metrics_do_not_become_zero(self):
        for values in ([], [None, 2], [float("nan"), 2], [float("inf"), 1]):
            with self.assertRaises(ValueError):
                normalize(values)
        with self.assertRaises(ValueError):
            harmonic([-1, 1])

    def test_harmonic_boundary_and_weights(self):
        self.assertEqual(harmonic([0, .8]), 0)
        self.assertAlmostEqual(harmonic([.9, .1]), .18)
        row = qualifications([{"candidate":"a", "G":.4,"U":.8,"P":.6,"S":.3}])[0]
        self.assertAlmostEqual(row["Q1"], 4/(1/.4+3/.8))
        self.assertAlmostEqual(row["Q2"], 6/(1/.4+1/.6+4/.3))
        self.assertLessEqual(row["Q3"], row["Q2"])

    def test_complete_scoring_and_capacity_cases(self):
        result = score_panel(read("qualification_inputs")["candidates"])
        expected = read("qualification_scores")["candidates"]
        for row, ref in zip(result, expected):
            for key in ("G", "U", "P", "S", "Psi", "Q1", "Q2", "Q3"):
                self.assertAlmostEqual(row[key], ref[key], places=12)
        self.assertAlmostEqual(result[0]["P"], .375)
        self.assertAlmostEqual(result[4]["U"], .9)
        self.assertEqual(assignment_summary(result)["optima"], [["M05", "M02", "M04"]])
        self.assertEqual(assignment_summary(result, {"M02":2})["optima"], [["M05", "M02", "M02"]])
        self.assertEqual(assignment_summary(result, roles=("Q1","Q2"))["optima"], [["M05", "M02"]])

    def test_infeasible_and_duplicate_assignment_inputs(self):
        row = {"candidate":"a", "Q1":.9, "Q2":.8, "Q3":.7}
        self.assertEqual(assignments([row]), [])
        with self.assertRaises(ValueError):
            assignment_summary([row])
        with self.assertRaises(ValueError):
            assignments([row, row])

    def test_normalization_and_gate_changes(self):
        report = diagnostics(read("qualification_inputs")["candidates"])
        self.assertEqual(report["normalization"]["endpoint"]["zero_q"],11)
        self.assertEqual(report["normalization"]["midpoint"]["zero_q"],0)
        self.assertEqual(len(report["gates"]["none"]["optima"]),2)
        self.assertEqual(report["gates"]["min"]["optima"],[["M05","M02","M01"]])
        change=report["leave_one_out"]["M03"]
        self.assertEqual([change["rank_comparison"][q]["reversals"] for q in ("Q1","Q2","Q3")],[1,2,1])
        self.assertNotEqual(change["scores"][0]["G"], read("qualification_scores")["candidates"][0]["G"])


class StatisticalTests(unittest.TestCase):
    def test_holm_uses_one_defined_family(self):
        expected = [.03,.06,.06]
        for a,b in zip(holm([.01,.04,.03]),expected):
            self.assertAlmostEqual(a,b)

    def test_paired_inference_uses_clusters(self):
        a=[{"record":i,"cluster":i//3,"cost":float(i+1)} for i in range(6)]
        b=[{"record":i,"cluster":i//3,"cost":1.0} for i in range(6)]
        result=paired_cluster_summary(a,b,"cost",replicates=500,seed=9)
        self.assertEqual(result["n_pairs"],6)
        self.assertEqual(result["team_statistics"]["df"],1)
        self.assertAlmostEqual(result["team_statistics"]["mean"],2.5)
        self.assertAlmostEqual(result["team_statistics"]["se"],1.5)
        self.assertEqual(result,paired_cluster_summary(a,b,"cost",replicates=500,seed=9))
        with self.assertRaises(ValueError):
            paired_cluster_summary(a,b[:-1],"cost")
        b[0]["cluster"]=9
        with self.assertRaises(ValueError):
            paired_cluster_summary(a,b,"cost")

    def test_mean_interval_and_missing_units(self):
        self.assertEqual(mean_interval([0,0,0])["ci"],[0,0])
        self.assertEqual(mean_interval([2,2,2])["p"],0)
        with self.assertRaises(ValueError):
            mean_interval([1])


class MetricTests(unittest.TestCase):
    def test_trajectory_and_fulfillment_units(self):
        self.assertEqual(planning_distance([1,4,3],[2,2,3]),3)
        self.assertEqual(variance_ratio([2,4,6],[1,2,3]),4)
        self.assertIsNone(variance_ratio([2,4,6],[1,1,1]))
        self.assertAlmostEqual(local_chasing([99,2,4,6],[1,2,3,4]),1)
        self.assertEqual(fulfillment_statistics([1,9],[1,0]),{"success":.5,"service":.1})
        self.assertEqual(normalized_chain_cost([10,2],[4,1]),5.5)

    def test_invalid_and_undefined_metrics(self):
        self.assertEqual(fulfillment_statistics([0,0],[0,0]),{"success":None,"service":None})
        with self.assertRaises(ValueError):
            fulfillment_statistics([1],[2])
        with self.assertRaises(ValueError):
            planning_distance([1],[1,2])
        with self.assertRaises(ValueError):
            normalized_chain_cost([1],[-1])


if __name__ == "__main__":
    unittest.main()
