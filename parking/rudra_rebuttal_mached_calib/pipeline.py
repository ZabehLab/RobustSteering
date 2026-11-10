"""Run or compare the isolated HarmBench rebuttal experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
from hashlib import file_digest
from pathlib import Path
import sys

import torch

from robust_steerability.artifacts import implementation_hash
from robust_steerability.benchmarks.execution import default_run_id, machine_provenance, utc_now
from robust_steerability.benchmarks.harmful_calibration import API_SCORERS, grid
from robust_steerability.benchmarks.launcher import run_jobs
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.judges.openai import score_generations as score_axbench
from robust_steerability.judges.specs import scorer_cache_path, scorer_spec

from .inputs import (
    MODEL_SPECS, REFERENCE_MODEL, UNIT, jacobian_worker, load_tensors, prepare_inputs, read_json, write_json,
)
from .reports import (
    TEST_ASR, VALIDATION_ASR, calibrate_scores, collect_test_scores, giga_results, giga_table,
    paired_results, response_text,
)
from .workers import (
    CALIBRATION_PROMPTS, CONDITIONS, EVAL_SHARDS, GIGA_METHODS, METHODS, PROFILES,
    candidate_path, classifier_worker, controller_identity,
    evaluation_split, generate_worker, generation_identity, generation_path, generation_root, task_inputs,
)

CACHE = UNIT / "cache"
MODULE = "parking.rudra_rebuttal_mached_calib.pipeline"
AXBENCH_BATCH_SIZE = 20


def calibration_tasks(profile: str) -> list[dict]:
    return [
        {"stage": "calibration", "profile": profile, "method": method, "configuration": config}
        for config in grid() for method in METHODS
    ]


def evaluation_tasks(winners: dict, part: str, profile: str) -> list[dict]:
    methods = GIGA_METHODS if profile == "giga" else METHODS
    return [
        {"stage": part, "profile": profile, "method": method, "condition": condition, "shard": shard,
         "configuration": None if method == "original" else winners[method]["configuration"]}
        for condition in CONDITIONS for method in methods for shard in range(EVAL_SHARDS)
    ]


def launch(kind: str, stage: str, model: str, devices: list[str], logs: Path, items: list, profile: str) -> None:
    if not items:
        return
    count = min(len(devices), len(items))
    jobs = []
    for index in range(count):
        name = f"{stage}-{kind}-{index:02d}"
        work = logs / f"{name}.json"
        write_json(work, {"kind": kind, "stage": stage, "profile": profile,
                          "model": model, "items": items[index::count]})
        jobs.append((name, [sys.executable, "-m", MODULE, "worker", str(work), "--device", "{device}"]))
    run_jobs(jobs, devices[:count], logs)


def run_worker(work: Path, device: str) -> None:
    torch.cuda.set_device(device)
    task = read_json(work)
    root = CACHE / task["model"]
    if task["kind"] == "generate":
        generate_worker(root, task["model"], device, task["items"])
    elif task["kind"] == "classify":
        classifier_worker(root, device, task["stage"], task["profile"], task["items"])
    elif task["kind"] == "jacobian":
        (index, count), = task["items"]
        jacobian_worker(root / "artifacts", MODEL_SPECS[task["model"]], device, index, count)
    else:
        raise ValueError(f"Unknown worker type: {task['kind']}")


@contextmanager
def run_log(model: str, profile: str, devices: list[str]):
    root = CACHE / model
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A run is already active for {model}") from exc
        logs = root / "logs" / default_run_id("matched-calib", model, profile)
        manifest = logs / "run.json"
        record = {
            "model": asdict(MODEL_SPECS[model]), "profile": profile, "devices": devices,
            "calibration_prompts_per_cell": CALIBRATION_PROMPTS["mega" if profile == "giga" else profile],
            "implementation_hash": implementation_hash(UNIT), "machine": machine_provenance(),
            "started_at_utc": utc_now(), "status": "running",
        }
        write_json(manifest, record)
        try:
            yield root, logs
        except BaseException as error:
            record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            raise
        else:
            record["status"] = "complete"
        finally:
            record["finished_at_utc"] = utc_now()
            write_json(manifest, record)


def seed_from_profile(root: Path, task: dict, controller: dict, records: list[dict],
                      batch_size: int, source_profile: str) -> int:
    """Copy verified generations and matching judge results to another profile."""
    stage = task["stage"]
    calibrating = stage == "calibration"
    source_task = {**task, "profile": source_profile}
    prefix = records[:CALIBRATION_PROMPTS["mini"]] if calibrating else records
    source = generation_path(root, source_task)
    if not source.exists():
        return 0
    saved = read_json(source)
    if saved["status"] != "complete" or saved["identity"] != generation_identity(controller, source_task, prefix, batch_size):
        return 0
    repetition, = saved["repetitions"]
    rows = repetition["rows"]
    prompt_ids = [row["prompt_id"] for row in prefix]
    if repetition["sample_count"] != len(prefix) or [row["prompt_id"] for row in rows] != prompt_ids:
        return 0

    destination = generation_path(root, task)
    identity = generation_identity(controller, task, records, batch_size)
    status = "partial" if calibrating else "complete"
    added = 0
    if destination.exists():
        current = read_json(destination)
        if current["identity"] != identity or current["repetitions"][0]["rows"][:len(rows)] != rows:
            return 0
    else:
        write_json(destination, {**saved, "identity": identity, "status": status, "reused_from": str(source)})
        added = len(rows)

    scorers = (VALIDATION_ASR, *API_SCORERS) if calibrating else (TEST_ASR,)
    for scorer in scorers:
        source_score = scorer_cache_path(generation_root(root, stage, source_profile), source, scorer)
        target_score = scorer_cache_path(generation_root(root, stage, task["profile"]), destination, scorer)
        if target_score.exists() or not source_score.exists():
            continue
        score = read_json(source_score)
        if score["status"] != "complete" or [row["prompt_id"] for row in score["rows"]] != prompt_ids:
            continue
        if any(not row.get("valid", True) for row in score["rows"]):
            continue
        spec = scorer_spec(scorer)
        if scorer in API_SCORERS:
            batches = score.get("batches", [])
            if (score["scorer"]["model"] != spec.model_id or score["scorer"]["batch_size"] != AXBENCH_BATCH_SIZE
                    or [batch["batch_index"] for batch in batches] != [0]
                    or [row["item_index"] for row in batches[0]["results"]] != list(range(len(prefix)))):
                continue
        elif score["scorer"]["model"] != [spec.model_id, spec.revision]:
            continue
        if calibrating:
            score.pop("completed_at_utc", None)
        write_json(target_score, {**score, "status": status, "reused_from": str(source_score)})
    return added


def generate(root: Path, model: str, stage: str, profile: str, tasks: list[dict],
             devices: list[str], logs: Path) -> list[Path]:
    inputs = root / "artifacts"
    fit = load_tensors(inputs / "reduced.pt")["fingerprint"]
    checkpoint = read_json(inputs / "manifest.json")["model"]
    tuning = read_json(inputs / "data.json")["calibration"]["tuning"]
    evaluation = read_json(inputs / "harmbench.json")
    expected = {}
    pending = []
    reused = 0
    for task in tasks:
        controller = None if task["method"] == "original" else candidate_path(root, task)
        path = generation_path(root, task)
        candidate_id = controller_identity(fit, checkpoint, task)
        records, batch_size = task_inputs(model, task, tuning, evaluation)
        expected[path] = generation_identity(candidate_id, task, records, batch_size)
        ready = controller is None or controller.exists()
        if controller is not None and ready:
            candidate = load_tensors(controller)
            if candidate["identity"] != candidate_id:
                raise ValueError(f"Controller identity mismatch: {controller}")
            if stage == "calibration" and candidate["status"] == "infeasible":
                continue
            if profile == "mega" and candidate["status"] == "complete":
                reused += seed_from_profile(root, task, candidate_id, records, batch_size, "mini")
            if profile == "giga" and stage == "screening" and candidate["status"] == "complete":
                reused += seed_from_profile(root, task, candidate_id, records, batch_size, "mega")
        if not ready or not path.exists() or read_json(path)["status"] != "complete":
            pending.append(task)
    if reused:
        print(f"{profile}/{stage}: reused {reused} cached responses", flush=True)
    launch("generate", stage, model, devices, logs, pending, profile)
    paths = []
    for task in tasks:
        if task["method"] != "original":
            candidate = load_tensors(candidate_path(root, task))
            if candidate["status"] == "infeasible":
                if stage == "calibration":
                    continue
                raise ValueError(f"Selected evaluation controller is infeasible: {task['method']}")
        path = generation_path(root, task)
        cached = read_json(path)
        if cached["identity"] != expected[path]:
            raise ValueError(f"Cached generation belongs to a different controller or split: {path}")
        if cached["status"] != "complete":
            raise ValueError(f"Missing completed generation: {path}")
        paths.append(path)
    return paths


def classify(root: Path, model: str, stage: str, profile: str, paths: list[Path],
             devices: list[str], logs: Path) -> None:
    scorer = VALIDATION_ASR if stage == "calibration" else TEST_ASR
    pending = []
    for path in paths:
        score = scorer_cache_path(generation_root(root, stage, profile), path, scorer)
        if not score.exists() or read_json(score)["status"] != "complete":
            pending.append(str(path))
    launch("classify", stage, model, devices, logs, pending, profile)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def run_giga(root: Path, model: str, devices: list[str], logs: Path) -> Path:
    output = root / "evaluation/giga"
    source = generation_root(root, "calibration", "mega") / "selection.json"
    selection = read_json(source)
    if selection["prompts_per_cell"] != CALIBRATION_PROMPTS["mega"] or set(selection["winners"]) != set(METHODS):
        raise ValueError(f"Giga requires a complete Mega selection: {source}")
    controllers = {
        method: candidate_path(root, {"method": method, "configuration": winner["configuration"]})
        for method, winner in selection["winners"].items()
    }
    provenance = {
        "selection": {"path": str(source), "sha256": file_sha256(source)},
        "controllers": {method: {"path": str(path), "sha256": file_sha256(path)}
                        for method, path in controllers.items()},
    }
    frozen = output / "calibration_source.json"
    if frozen.exists():
        if read_json(frozen) != provenance:
            raise ValueError("Giga's frozen Mega selection or controller files changed")
    else:
        write_json(frozen, provenance)

    evaluation = read_json(root / "artifacts/harmbench.json")
    parts = ("screening", "holdout")
    records = {condition: [row for part in parts for row in evaluation_split(evaluation, condition, part)]
               for condition in CONDITIONS}
    direct, jailbreak = records["direct"], records["human_jailbreak"]
    behaviors = {row["behavior_id"] for row in direct}
    variants = Counter((row["behavior_id"], row["template_index"]) for row in jailbreak)
    if len(behaviors) != len(direct) or variants != {(behavior, i): 1 for behavior in behaviors for i in range(5)}:
        raise ValueError("Giga requires one direct prompt and all five matched jailbreak variants per behavior")

    all_tasks, paths_by_part = [], {}
    for part in parts:
        tasks = evaluation_tasks(selection["winners"], part, "giga")
        paths = generate(root, model, part, "giga", tasks, devices, logs)
        classify(root, model, part, "giga", paths, devices, logs)
        all_tasks.extend(tasks)
        paths_by_part[part] = paths
    for part, paths in paths_by_part.items():
        print(f"giga/{part}: scoring AXBench on held-out responses", flush=True)
        score_axbench(paths, generation_root(root, part, "giga"), list(API_SCORERS),
                      concurrency=32, batch_size=AXBENCH_BATCH_SIZE)

    asr = collect_test_scores(root, all_tasks)
    axbench = [collect_test_scores(root, all_tasks, scorer) for scorer in API_SCORERS]
    results = giga_results(records, asr, axbench)
    write_json(output / "results.json", {
        "model": model, "profile": "giga", "behavior_count": len(direct), "scorer": TEST_ASR,
        "selection": selection, "calibration_source": provenance, "results": results,
        "run_log": str(logs / "run.json"),
    })
    return write_response(output, response_text(MODEL_SPECS[model].label, "giga", selection, results, len(direct)))


def write_response(output: Path, text: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    response = output / "response.md"
    temporary = response.with_suffix(".md.tmp")
    temporary.write_text(text)
    temporary.replace(response)
    return response


def run_experiment(model: str, profile: str, devices: list[str]) -> Path:
    with run_log(model, profile, devices) as (root, logs):
        if profile == "giga":
            return run_giga(root, model, devices, logs)
        prepare_inputs(
            root / "artifacts", MODEL_SPECS[model], devices,
            lambda workers: launch("jacobian", "artifacts", model, workers, logs,
                                   [[i, len(workers)] for i in range(len(workers))], profile),
        )
        tasks = calibration_tasks(profile)
        calibration_root = generation_root(root, "calibration", profile)
        paths = generate(root, model, "calibration", profile, tasks, devices, logs)
        if not paths:
            raise RuntimeError("No feasible calibration generations")
        classify(root, model, "calibration", profile, paths, devices, logs)
        score_axbench(paths, calibration_root, list(API_SCORERS), concurrency=32, batch_size=AXBENCH_BATCH_SIZE)
        selection = calibrate_scores(root, profile, tasks, set(paths))
        parts = ["screening", "holdout"] if profile == "full" else ["screening"]
        evaluation = read_json(root / "artifacts/harmbench.json")
        all_tasks = []
        for part in parts:
            test_tasks = evaluation_tasks(selection["winners"], part, profile)
            paths = generate(root, model, part, profile, test_tasks, devices, logs)
            classify(root, model, part, profile, paths, devices, logs)
            all_tasks.extend(test_tasks)

        scores = collect_test_scores(root, all_tasks)
        records = {
            condition: [row for part in parts for row in evaluation_split(evaluation, condition, part)]
            for condition in CONDITIONS
        }
        results = paired_results(records, scores)
        summary = {
            "model": model, "profile": profile, "behavior_count": len(records["direct"]),
            "scorer": TEST_ASR, "selection": selection, "results": results, "run_log": str(logs / "run.json"),
        }
        if profile == "full":
            holdout = {condition: evaluation_split(evaluation, condition, "holdout") for condition in CONDITIONS}
            tail_scores = collect_test_scores(root, [task for task in all_tasks if task["stage"] == "holdout"])
            summary["holdout_after_screening"] = paired_results(holdout, tail_scores)
        output = root / "evaluation" / profile
        response = write_response(output, response_text(MODEL_SPECS[model].label, profile, selection, results, len(records["direct"])))
        write_json(output / "results.json", summary)
    return response


def compare(profile: str) -> None:
    paths = sorted(CACHE.glob(f"*/evaluation/{profile}/results.json"))
    if not paths:
        print(f"No completed {profile} runs.")
        return
    if profile == "giga":
        for path in paths:
            summary = read_json(path)
            print(f"{MODEL_SPECS[summary['model']].label} ({summary['behavior_count']} behaviors)")
            print("\n".join(giga_table(summary["selection"]["winners"], summary["results"])))
        return
    print("| Model | Behaviors | A-LQR (Full-width) ASR % | A-LQR (8D) ASR % | H∞ ASR % | Full-width gap (pp) | 8D gap (pp) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for path in paths:
        summary = read_json(path)
        asr = [
            summary["results"][method]["human_jailbreak"]["asr_percent"]["full_sample_mean"]
            for method in METHODS
        ]
        values = [*(f"{value:.2f}" for value in asr), f"{asr[0] - asr[2]:+.2f}", f"{asr[1] - asr[2]:+.2f}"]
        print(f"| {summary['model']} | {summary['behavior_count']} | " + " | ".join(values) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--model", choices=MODEL_SPECS, default=REFERENCE_MODEL)
    run.add_argument("--profile", choices=PROFILES, default="mini")
    run.add_argument("--devices", default="auto")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--profile", choices=PROFILES, default="mini")
    worker = commands.add_parser("worker")
    worker.add_argument("work", type=Path)
    worker.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.command == "run":
        print(run_experiment(args.model, args.profile, resolve_cuda_devices(args.devices)))
    elif args.command == "compare":
        compare(args.profile)
    else:
        run_worker(args.work, args.device)


if __name__ == "__main__":
    main()
