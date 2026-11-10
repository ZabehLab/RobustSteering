"""Focused controller and pipeline checks for the matched-calibration unit."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from robust_steerability.benchmarks.methods import method_spec
from robust_steerability.control import FiniteHorizonControlProblem, LQRController
from robust_steerability.experiments import methods
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec

from . import pipeline, workers
from .inputs import REFERENCE_MODEL, read_json, write_json, write_torch
from .reports import TEST_ASR, VALIDATION_ASR


class ReducedLQRTests(unittest.TestCase):
    def test_registered_policy_applies_signed_gain_in_matched_coordinates(self):
        torch.manual_seed(10)
        basis, _ = torch.linalg.qr(torch.randn(12, 8))
        problem = FiniteHorizonControlProblem(
            dynamics=torch.eye(8)[None] * 0.8,
            control_channels=torch.eye(8)[None] * 0.7,
            state_costs=torch.eye(8)[None],
            control_costs=torch.eye(8)[None],
            terminal_cost=torch.eye(8),
        )
        solution = LQRController.synthesize(problem).solution()
        artifact = methods.ReducedLQRArtifact(
            gains=solution.gains,
            control_channels=problem.control_channels,
            means=torch.randn(1, 12),
            encoders=basis[None],
            decoders=basis[None],
            feature_unit=torch.eye(8)[0][None],
            setpoints=torch.tensor([0.25]),
        )
        self.assertEqual(method_spec("alqr_8d").label, "A-LQR (8D)")
        policy = methods.build_policy("alqr_8d", artifact, kp=0, ki=0, kd=0)
        policy.prepare(torch.device("cpu"), torch.float32)
        states = torch.randn(3, 12)
        error = (states - artifact.means[0]) @ basis
        error[:, 0] -= 0.25
        expected = error @ solution.gains[0].T @ problem.control_channels[0].T @ basis.T
        torch.testing.assert_close(policy.activation_delta(0, states), expected)

    def test_full_width_method_keeps_its_registry_entry(self):
        self.assertEqual(method_spec("alqr").label, "A-LQR")


class PipelineTests(unittest.TestCase):
    def test_mini_uses_the_same_inputs_across_models_and_keeps_full_separate(self):
        direct = [
            {"prompt_id": f"b{i}-direct", "behavior_id": f"b{i}",
             "functional_category": "standard", "semantic_category": "fixture", "text": "fixture"}
            for i in range(240)
        ]
        evaluation = {"evaluation": {
            "direct": direct,
            "human_jailbreak": [
                {**row, "prompt_id": f"{row['behavior_id']}-human-{j}", "template_index": j}
                for row in direct for j in range(5)
            ],
        }}
        tuning = [{"prompt_id": f"val-{i}"} for i in range(50)]
        calls = Counter()

        def prepare(root, spec, _devices, _launch):
            write_json(root / "manifest.json", {"model": [spec.model_id, spec.revision]})
            write_torch(root / "reduced.pt", {"fingerprint": "fixture"})
            write_json(root / "data.json", {"calibration": {"tuning": tuning}})
            write_json(root / "harmbench.json", evaluation)

        def score(root, path, scorer, score_value):
            rows = read_json(path)["repetitions"][0]["rows"]
            spec = scorer_spec(scorer)
            data = {
                "status": "complete",
                "rows": [{"prompt_id": row["prompt_id"], "score": score_value(row)} for row in rows],
                "scorer": {"model": [spec.model_id, spec.revision]},
            }
            if spec.backend == "openai_0_2":
                data["scorer"] = {"model": spec.model_id, "batch_size": 20}
                data["batches"] = [{"batch_index": start // 20, "results": [
                    {"item_index": index} for index in range(start, min(start + 20, len(rows)))
                ]} for start in range(0, len(rows), 20)]
            write_json(scorer_cache_path(root, path, scorer), data)

        def launch(kind, stage, model, devices, _logs, items, profile):
            self.assertEqual(devices, ["cuda:0", "cuda:1"])
            calls[kind, profile, stage] += len(items)
            root = pipeline.CACHE / model
            if kind == "classify":
                for item in items:
                    path = Path(item)
                    method = read_json(path)["identity"]["method"]
                    test_value = {
                        "original": lambda row: 1,
                        "alqr": lambda row: 1,
                        "alqr_8d": lambda row: int(row["prompt_id"].split("-")[0][1:]) % 2,
                        "h_infinity": lambda row: 0,
                    }[method]
                    key = VALIDATION_ASR if stage == "calibration" else TEST_ASR
                    score_root = workers.generation_root(root, stage, profile)
                    if stage == "calibration" and profile == "mega":
                        cached = read_json(scorer_cache_path(score_root, path, key))
                        self.assertEqual((cached["status"], len(cached["rows"])), ("partial", 20))
                    score(score_root, path, key, (lambda _row: 0) if stage == "calibration" else test_value)
                return
            self.assertEqual(kind, "generate")
            model_id = read_json(root / "artifacts/manifest.json")["model"]
            for task in items:
                cid = workers.controller_identity("fixture", model_id, task)
                if stage == "calibration":
                    infeasible = task["method"] == "h_infinity" and task["configuration"]["q"] == 10
                    write_torch(workers.candidate_path(root, task), {
                        "identity": cid, "status": "infeasible" if infeasible else "complete",
                    })
                    if infeasible:
                        continue
                records, batch = workers.task_inputs(model, task, tuning, evaluation)
                destination = workers.generation_path(root, task)
                if stage == "calibration":
                    self.assertEqual(records, tuning[:20] if profile == "mini" else tuning)
                    if profile == "mega":
                        seeded = read_json(destination)
                        self.assertEqual(seeded["status"], "partial")
                        self.assertEqual(seeded["repetitions"][0]["sample_count"], 20)
                else:
                    behaviors = direct[:20] if stage == "screening" else direct[20:]
                    expected_ids = {row["behavior_id"] for row in behaviors}
                    expected = [row for row in evaluation["evaluation"][task["condition"]]
                                if row["behavior_id"] in expected_ids]
                    if profile == "giga" and stage == "holdout":
                        expected.sort(key=lambda row: row["prompt_id"])
                    self.assertEqual(records, expected[task["shard"]::2])
                write_json(destination, {
                    "identity": workers.generation_identity(cid, task, records, batch),
                    "status": "complete",
                    "repetitions": [{"repetition": 0, "sample_count": len(records),
                                     "rows": [{"prompt_id": row["prompt_id"], "control_energy": 1.0}
                                              for row in records]}],
                })

        def axbench(paths, root, scorers, **_kwargs):
            for path in paths:
                if root.parent.name == "giga":
                    for scorer in scorers:
                        score(root, path, scorer, lambda _row: 1)
                    continue
                config = read_json(path)["identity"]["configuration"]
                reuse_winner = root.name == "mega" and path.parent.name != "alqr"
                winner = "q_00_qf_00" if root.name == "mini" or reuse_winner else "q_01_qf_00"
                for scorer in scorers:
                    if root.name == "mega":
                        cached = read_json(scorer_cache_path(root, path, scorer))
                        self.assertEqual((cached["status"], len(cached["batches"])), ("partial", 1))
                    score(root, path, scorer, lambda _row: 2 if config["grid_id"] == winner else 1)

        with TemporaryDirectory() as tmp, patch.multiple(
            pipeline, CACHE=Path(tmp), prepare_inputs=prepare, launch=launch,
            score_axbench=axbench, machine_provenance=lambda: {},
        ):
            mini = pipeline.run_experiment(REFERENCE_MODEL, "mini", ["cuda:0", "cuda:1"])
            pipeline.run_experiment("smollm2_360m_instruct", "mini", ["cuda:0", "cuda:1"])
            mega = pipeline.run_experiment(REFERENCE_MODEL, "mega", ["cuda:0", "cuda:1"])
            mega_report = mega.read_bytes()
            mega_selection = pipeline.CACHE / REFERENCE_MODEL / "calibration/mega/selection.json"
            mega_data = mega_selection.read_bytes()
            giga = pipeline.run_experiment(REFERENCE_MODEL, "giga", ["cuda:0", "cuda:1"])
            self.assertEqual(mega.read_bytes(), mega_report)
            self.assertEqual(mega_selection.read_bytes(), mega_data)
            full = pipeline.run_experiment(REFERENCE_MODEL, "full", ["cuda:0", "cuda:1"])
            pipeline.run_experiment(REFERENCE_MODEL, "mini", ["cuda:0", "cuda:1"])
            self.assertEqual(calls["generate", "mini", "calibration"], 72)
            self.assertEqual(calls["generate", "mega", "calibration"], 33)
            self.assertEqual(calls["generate", "mega", "screening"], 4)
            self.assertEqual(calls["classify", "mega", "screening"], 4)
            self.assertEqual(calls["generate", "mega", "holdout"], 0)
            self.assertEqual(calls["generate", "giga", "calibration"], 0)
            self.assertEqual(calls["generate", "giga", "screening"], 4)
            self.assertEqual(calls["classify", "giga", "screening"], 4)
            self.assertEqual(calls["generate", "giga", "holdout"], 16)
            self.assertEqual(calls["generate", "full", "calibration"], 33)
            self.assertEqual(calls["generate", "mini", "screening"], 24)
            self.assertEqual(calls["generate", "full", "screening"], 12)
            self.assertEqual(calls["generate", "full", "holdout"], 12)
            self.assertEqual(calls["classify", "mini", "screening"], 24)
            mini_result = read_json(mini.with_name("results.json"))
            self.assertEqual(mini_result["behavior_count"], 20)
            self.assertEqual(mini_result["selection"]["prompts_per_cell"], 20)
            self.assertIn("20 validation responses per cell", mini.read_text())
            self.assertIn("| A-LQR (8D) | (0.01, 0.01) | 50.00 | 50.00 |", mini.read_text())
            mega_result = read_json(mega.with_name("results.json"))
            self.assertEqual(mega_result["behavior_count"], 20)
            self.assertEqual(mega_result["selection"]["prompts_per_cell"], 50)
            self.assertNotIn("holdout_after_screening", mega_result)
            giga_result = read_json(giga.with_name("results.json"))
            self.assertEqual(giga_result["behavior_count"], 240)
            self.assertEqual(giga_result["selection"], mega_result["selection"])
            self.assertEqual(giga_result["results"]["original"]["axbench_overall"]["full_sample_mean"], 1)
            self.assertIn("| Original | — | 100.00 ± 0.00 | 100.00 ± 0.00 |\n", giga.read_text())
            self.assertNotIn("AXBench Overall (0–2)", giga.read_text())
            self.assertNotIn("gap vs H∞", giga.read_text())
            result = read_json(full.with_name("results.json"))
            self.assertEqual(result["behavior_count"], 240)
            self.assertEqual(result["selection"]["prompts_per_cell"], 50)
            self.assertIn("holdout_after_screening", result)
            self.assertEqual(result["results"]["alqr_8d"]["human_jailbreak"]["asr_percent"]["full_sample_mean"], 50)
            self.assertIn("| 10 | 1/√10 | 1.000 | 1.000 | infeasible |", full.read_text())
            self.assertIn("| A-LQR (8D) | (0.1, 0.01) | 50.00 | 50.00 |", full.read_text())

    def test_test_classifier_uses_the_memory_safe_batch_size(self):
        root = Path("model-cache")
        path = root / "evaluation/mini/screening/generations/direct/alqr/shard-0.json"
        with patch.object(workers, "score_generations") as score, \
                patch.object(workers, "load_access_token", return_value="fixture"):
            workers.classifier_worker(root, "cuda:0", "screening", "mini", [str(path)])
            score.assert_called_once_with(
                [path], root / "evaluation/mini/screening", TEST_ASR, "cuda:0", "fixture", batch_size=1,
            )

    def test_batch_resume_keeps_completed_prompt_prefix(self):
        task = {"method": "alqr_8d", "stage": "screening", "profile": "mini", "condition": "direct", "shard": 0}
        records = [
            {"prompt_id": str(i), "behavior_id": str(i), "behavior": "", "context": "",
             "functional_category": "", "semantic_category": "", "text": "", "model_input": str(i)}
            for i in range(3)
        ]
        identity = {"fixture": True}
        with TemporaryDirectory() as tmp, patch.object(workers, "generate_completions") as generation:
            root = Path(tmp)
            path = workers.generation_path(root, task)
            generation.side_effect = [(["first", "second"], [1, 1]), RuntimeError("fixture interruption")]
            with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                workers.run_generation_task(root, task, records, identity, None, None, None, 2)
            self.assertEqual(read_json(path)["status"], "partial")
            generation.side_effect = None
            generation.return_value = (["third"], [1])
            workers.run_generation_task(root, task, records, identity, None, None, None, 2)
            self.assertEqual(generation.call_args.args[2], ["2"])
            rows = read_json(path)["repetitions"][0]["rows"]
            self.assertEqual([row["completion"] for row in rows], ["first", "second", "third"])


if __name__ == "__main__":
    unittest.main()
