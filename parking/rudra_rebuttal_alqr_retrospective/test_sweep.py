import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import sweep


class SweepTest(unittest.TestCase):
    def test_archived_inputs_match_paper(self):
        for model in sweep.MODELS:
            records, cells = sweep.paper_inputs(model)
            self.assertEqual(len(records), 50)
            self.assertEqual(len(cells), 12)
            self.assertTrue(all(cell["r"] == 1 and cell["lambda"] == 1.5 for cell in cells))

    def test_resumes_completed_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cell.json"
            identity = {"method": "alqr", "generation_batch_size": 2}
            records = [dict(prompt_id=str(i), text=f"formatted {i}", raw_request=f"raw {i}",
                            behavior_id=str(i), behavior="", context="", functional_category="",
                            semantic_category="") for i in range(3)]
            with patch.object(sweep, "generate_completions", side_effect=[(["a", "b"], [1, 1]), RuntimeError("interrupted")]):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    sweep.generate_cell(path, identity, records, None, None, None)
            with patch.object(sweep, "generate_completions", return_value=(["c"], [1])) as generate:
                sweep.generate_cell(path, identity, records, None, None, None)
                sweep.generate_cell(path, identity, records, None, None, None)
            generate.assert_called_once()
            self.assertEqual(generate.call_args.args[2], ["formatted 2"])
            saved = sweep.load_json(path)
            self.assertEqual(saved["status"], "complete")
            self.assertEqual([row["text"] for row in saved["repetitions"][0]["rows"]], ["raw 0", "raw 1", "raw 2"])


if __name__ == "__main__":
    unittest.main()
