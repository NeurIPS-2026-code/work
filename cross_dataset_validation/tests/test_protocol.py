from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path

from validation.common import load_config, stable_hash


ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_config_points_to_conv26_at_index_zero(self):
        config = load_config(ROOT / "config.example.json")
        with Path(config["paths"]["locomo_data"]).open(
            "r",
            encoding="utf-8",
        ) as handle:
            samples = json.load(handle)
        self.assertEqual(samples[0]["sample_id"], "conv-26")
        self.assertEqual(len(samples[0]["qa"]), 199)
        self.assertEqual(
            sum(question["category"] != 5 for question in samples[0]["qa"]),
            152,
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

