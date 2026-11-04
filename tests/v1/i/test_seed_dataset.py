"""只验证种子数据契约，不运行 Agent、模型或商城。"""
import ast
from collections import Counter
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
CASES = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
CATEGORIES = {"商品", "政策", "订单", "人工", "澄清", "拒答"}


class SeedDatasetTests(unittest.TestCase):
    def test_schema_and_behavior_contract(self):
        for case in CASES:
            with self.subTest(case=case["id"]):
                self.assertRegex(case["id"], r"^V1-I-\d{2}-\d{2}$")
                self.assertIn(case["category"], CATEGORIES)
                for key in ("source", "split", "paraphrase_group", "difficulty", "identity", "initial_state"):
                    self.assertIsInstance(case[key], str)
                    self.assertTrue(case[key].strip())
                for key in ("requirements", "tags", "fixture_requirements"):
                    self.assertTrue(case[key])
                    self.assertTrue(all(isinstance(value, str) and value.strip() for value in case[key]))
                self.assertTrue(case["user_turns"])
                for turn in case["user_turns"]:
                    self.assertEqual(set(turn), {"user"})
                    self.assertIsInstance(turn["user"], str)
                    self.assertTrue(turn["user"].strip())
                    self.assertLessEqual(len(turn["user"]), 6000)
                self.assertEqual("multi_turn" in case["tags"], len(case["user_turns"]) > 1)
                expected = case["expected"]
                for key in ("behavior", "termination", "scoring_note"):
                    self.assertIsInstance(expected[key], str)
                    self.assertTrue(expected[key].strip())
                for key in ("allowed_final_intents", "forbidden", "checks"):
                    self.assertTrue(expected[key])
                    self.assertTrue(all(isinstance(value, str) and value.strip() for value in expected[key]))
                plan = expected["allowed_tool_plan"]
                self.assertIn(plan["kind"], ("current_functions", "future_semantics"))
                for key in ("steps", "constraints"):
                    self.assertTrue(plan[key])
                    self.assertTrue(all(isinstance(value, str) and value.strip() for value in plan[key]))
                self.assertIsInstance(case["safety_hard_gate"], bool)

    def test_size_coverage_and_unique_scenarios(self):
        self.assertGreaterEqual(len(CASES), 120)
        self.assertEqual(len({case["id"] for case in CASES}), len(CASES))
        scenarios = {(tuple(turn["user"] for turn in case["user_turns"]), tuple(case["fixture_requirements"])) for case in CASES}
        self.assertEqual(len(scenarios), len(CASES))
        counts = Counter(case["category"] for case in CASES)
        self.assertEqual(set(counts), CATEGORIES)
        self.assertTrue(all(count >= 20 for count in counts.values()))
        self.assertGreaterEqual(sum(len(case["user_turns"]) > 1 for case in CASES), 40)
        self.assertGreaterEqual(sum("adversarial" in case["tags"] for case in CASES), 20)

    def test_readiness_does_not_claim_execution_or_missing_business(self):
        for case in CASES:
            with self.subTest(case=case["id"]):
                readiness = case["readiness"]
                self.assertIs(readiness["runnable_now"], False)
                self.assertEqual(readiness["execution_status"], "not_run")
                self.assertTrue(readiness["blockers"])
                self.assertIn("runner", " ".join(readiness["blockers"]))
                missing_business = case["category"] in {"政策", "订单", "人工"}
                self.assertEqual(readiness["business_available"], not missing_business)
                if missing_business:
                    self.assertTrue(any(value.startswith("K-") for value in readiness["blockers"]))
                    self.assertEqual(case["expected"]["allowed_tool_plan"]["kind"], "future_semantics")
                self.assertEqual(case["split"], "development_seed")
                self.assertIn("人工构造", case["source"])
                self.assertEqual(case["versions"], {"dataset": "v1-seed-1", "prompt_hash": None, "model_deployment": None, "fixture_hash": None})

    def test_current_function_references_exist_without_importing_agent(self):
        references = {step for case in CASES if case["expected"]["allowed_tool_plan"]["kind"] == "current_functions"
                      for step in case["expected"]["allowed_tool_plan"]["steps"]}
        for reference in references:
            with self.subTest(reference=reference):
                module, name = reference.rsplit(".", 1)
                path = ROOT / (module.replace(".", "/") + ".py")
                self.assertTrue(path.is_file(), path)
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                self.assertIn(name, {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))})


if __name__ == "__main__":
    unittest.main()
