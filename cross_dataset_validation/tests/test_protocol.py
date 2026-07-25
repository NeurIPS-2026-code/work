from __future__ import annotations

import runpy
import unittest
from pathlib import Path

from validation.common import load_config, stable_hash


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_server_layout_and_vendored_core(self):
        config = load_config(ROOT / "config.example.json")
        self.assertEqual(
            config["data"]["locomo_source_sample_id"],
            "conv-26",
        )
        self.assertFalse(config["data"]["download_if_missing"])
        self.assertEqual(
            Path(config["paths"]["locomo_data"]).parts[-2:],
            ("locomo", "locomo10.json"),
        )
        self.assertEqual(
            Path(config["paths"]["hotpotqa_data"]).parts[-2:],
            ("hotpotqa", "eval_400.json"),
        )
        source = Path(config["paths"]["source_root"])
        self.assertTrue((source / "gam" / "__init__.py").is_file())
        self.assertTrue((source / "eval" / "locomo_test.py").is_file())
        self.assertTrue(
            (source / "exp" / "eval" / "research_agent_exp_main.py").is_file()
        )

    def test_composite_conditions_follow_original_code(self):
        module = runpy.run_path(str(ROOT / "02_build_experience_bank.py"))
        trace = {
            "question": "Who did X?",
            "integrated_memory": "answer",
            "iterations": [
                {
                    "step": 0,
                    "plan": {"tools": ["vector"]},
                    "temp_memory": {"content": "partial"},
                    "decision": {
                        "enough": False,
                        "new_request": "Find Y",
                    },
                },
                {
                    "step": 1,
                    "plan": {"tools": ["keyword"]},
                    "temp_memory": {"content": "complete"},
                    "decision": {"enough": True, "new_request": None},
                },
            ],
        }
        question, _, condition = module["step_context"](
            trace,
            1,
            "Planning",
        )
        self.assertEqual(question, "Find Y")
        self.assertEqual(condition, "Find Y")
        _, _, condition = module["step_context"](trace, 1, "Reflection")
        self.assertEqual(
            condition,
            "question:Who did X? temp_memory:complete",
        )

    def test_bootstrap_is_seeded(self):
        module = runpy.run_path(str(ROOT / "04_summarize.py"))
        first = module["paired_bootstrap_ci"](
            [0.1, -0.1, 0.2],
            100,
            2026,
        )
        second = module["paired_bootstrap_ci"](
            [0.1, -0.1, 0.2],
            100,
            2026,
        )
        self.assertEqual(first, second)

    def test_protocol_hash_is_order_independent(self):
        self.assertEqual(
            stable_hash({"a": 1, "b": 2}),
            stable_hash({"b": 2, "a": 1}),
        )


if __name__ == "__main__":
    unittest.main()
