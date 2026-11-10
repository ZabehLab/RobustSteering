"""Run the submitted full-width A-LQR on the archived HarmBench cost grid."""

from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from hashlib import file_digest
from pathlib import Path
from sys import executable
from time import perf_counter

import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.benchmarks.harmful_artifacts import _save_torch, _write_json, load_model
from robust_steerability.benchmarks.harmful_runtime import MAX_NEW_TOKENS, generate_completions
from robust_steerability.benchmarks.launcher import run_jobs
from robust_steerability.control import LQRController
from robust_steerability.datasets.harmful import SAFE_CONCEPT
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.judges.openai import score_generations
from robust_steerability.runtime.policy import SemanticSetpointPolicy
from robust_steerability.source_methods.control import SetpointCalibration, build_alqr_policy

from . import audit
from .audit import MODELS, OUTPUT, PAPER_CACHE, RUBRICS, load_json, prompt_inputs, read_generations


PARAMETERS = ("q", "r", "q_final", "lambda")
MODULE = "parking.rudra_rebuttal_alqr_retrospective.sweep"


def paper_inputs(model_key: str) -> tuple[list[dict], list[dict]]:
    root = PAPER_CACHE / model_key
    archived = load_json(root / "artifacts/data.json")
    records = archived["calibration"]["tuning"]
    selected = root / "calibrations/h_infinity/selected"
    manifest = load_json(selected / "selection.json")
    expected = {row["prompt_id"]: (row["behavior_id"], row["raw_request"]) for row in records}
    if archived["model"] != manifest["model"] or len(expected) != len(records) or len(records) != 50:
        raise ValueError(f"Invalid archived model or validation records for {model_key}")
    cells = []
    for cell in manifest["grid"]:
        identity, generations = read_generations(selected / "grid/generations" / f"{cell['grid_id']}.json")
        if (identity["model"] != archived["model"] or identity["evaluated_model_kv_cache"]
            or prompt_inputs(generations) != expected or list(generations) != list(expected)
            or any(identity["configuration"][key] != cell[key] for key in PARAMETERS)):
            raise ValueError(f"Archived H∞ inputs differ for {model_key}/{cell['grid_id']}")
        cells.append({"grid_id": cell["grid_id"], **{key: cell[key] for key in PARAMETERS},
                      "paper_batch_size": identity["generation_batch_size"]})
    if len(cells) != 12 or len({cell["grid_id"] for cell in cells}) != 12:
        raise ValueError(f"Invalid paper grid for {model_key}")
    return records, cells


def generation_payload(path: Path, identity: dict, records: list[dict]) -> dict:
    if not path.exists():
        return {"identity": identity, "status": "partial", "repetitions": [{"repetition": 0, "rows": []}]}
    data = load_json(path)
    rows = data["repetitions"][0]["rows"]
    if data["identity"] != identity or [row["prompt_id"] for row in rows] != [
        row["prompt_id"] for row in records[:len(rows)]
    ] or (data["status"] == "complete") != (len(rows) == len(records)):
        raise ValueError(f"Saved responses do not match frozen inputs in {path}")
    return data


def generate_cell(path: Path, identity: dict, records: list[dict], model, tokenizer, policy) -> None:
    data = generation_payload(path, identity, records)
    repetition = data["repetitions"][0]
    rows = repetition["rows"]
    batch_size = identity["generation_batch_size"]
    while len(rows) < len(records):
        batch = records[len(rows):len(rows) + batch_size]
        started = perf_counter()
        completions, counts = generate_completions(
            model, tokenizer, [row["text"] for row in batch], steering_policy=policy,
            use_cache=False, batch_size=len(batch), max_new_tokens=MAX_NEW_TOKENS,
        )
        for row, completion, count in zip(batch, completions, counts, strict=True):
            saved = {key: row[key] for key in ("prompt_id", "behavior_id", "behavior", "context",
                                               "functional_category", "semantic_category")}
            saved.update(text=row["raw_request"], concept=SAFE_CONCEPT, completion=completion, generated_tokens=count)
            rows.append(saved)
        repetition["sample_count"] = len(rows)
        data["status"] = "complete" if len(rows) == len(records) else "partial"
        _write_json(path, data)
        print(f"{path.stem}: saved {len(rows)}/{len(records)} responses ({perf_counter() - started:.1f}s)", flush=True)


def worker(model_key: str, device: str, shard: int, shards: int) -> None:
    records, cells = paper_inputs(model_key)
    artifacts = PAPER_CACHE / model_key / "artifacts"
    archived = load_json(artifacts / "data.json")
    inputs = {}
    for filename in ("dynamics.pt", "setpoint.pt"):
        with (artifacts / filename).open("rb") as handle:
            inputs[filename] = file_digest(handle, "sha256").hexdigest()
    grid = OUTPUT / model_key / "grid"
    pending = []
    for cell in cells[shard::shards]:
        params = {key: cell[key] for key in PARAMETERS}
        controller_id = {"model": archived["model"], "method": "alqr", "parameters": params, "artifacts": inputs}
        identity = {**controller_id, "grid_id": cell["grid_id"], "prompt_hash": configuration_hash({"rows": records}),
                    "evaluated_model_kv_cache": False, "max_new_tokens": MAX_NEW_TOKENS,
                    "generation_batch_size": min(16, cell["paper_batch_size"]) if model_key == MODELS[-1] else cell["paper_batch_size"]}
        path = grid / "generations" / f"{cell['grid_id']}.json"
        if generation_payload(path, identity, records)["status"] != "complete":
            pending.append((path, controller_id, identity))
    if not pending:
        return

    model, tokenizer = load_model(model_key, device)
    dynamics = torch.load(artifacts / "dynamics.pt", map_location="cpu", weights_only=True, mmap=True)["dynamics"]
    setpoint_payload = torch.load(artifacts / "setpoint.pt", map_location="cpu", weights_only=True)
    setpoint = SetpointCalibration(setpoint_payload["contrast"], setpoint_payload["feature_norm"])
    for path, controller_id, identity in pending:
        params = controller_id["parameters"]
        controller_path = grid / "controllers" / f"{path.stem}.pt"
        if controller_path.exists():
            saved = torch.load(controller_path, map_location="cpu", weights_only=True)
            if saved["identity"] != controller_id:
                raise ValueError(f"Saved gains do not match frozen inputs in {controller_path}")
            policy = SemanticSetpointPolicy(LQRController(saved["gains"]), setpoint.unit_features(0.0),
                                            setpoint.setpoints(params["lambda"]))
            del saved
        else:
            started = perf_counter()
            policy = build_alqr_policy(dynamics, setpoint, multiplier=params["lambda"], q=params["q"],
                                       r=params["r"], q_final=params["q_final"], device=device)
            _save_torch(controller_path, {"identity": controller_id, "gains": policy.controller.solution().gains})
            print(f"{path.stem}: cached full-width controller ({perf_counter() - started:.1f}s)", flush=True)
        generate_cell(path, identity, records, model, tokenizer, policy)
        del policy


def score_ready(model_key: str, scored: set[Path]) -> int:
    root = OUTPUT / model_key / "grid"
    ready = sorted(path for path in (root / "generations").glob("*.json")
                   if path not in scored and load_json(path)["status"] == "complete")
    if ready:
        print(f"{model_key}: scoring {len(ready)} completed cells with AXBench", flush=True)
        score_generations(ready, root, list(RUBRICS), concurrency=32, batch_size=20)
        scored.update(ready)
    return len(ready)


def run(devices: list[str]) -> None:
    # Start one 8B and one 1B worker, then fill available slots with the remaining shards.
    jobs = [(MODELS[-1], 0, len(devices)), (MODELS[0], 0, 1)]
    jobs.extend((MODELS[-1], shard, len(devices)) for shard in range(1, len(devices)))
    jobs.extend((MODELS[1], shard, len(devices)) for shard in range(len(devices)))
    commands = [(f"{key}-shard-{shard}", [executable, "-u", "-m", MODULE, "worker", key,
                 "{device}", str(shard), str(shards)]) for key, shard, shards in jobs]
    logs = OUTPUT / "logs"
    _write_json(logs / "run.json", {"devices": devices, "jobs": jobs, "responses": 1800,
                                   "scorers": list(RUBRICS), "max_new_tokens": MAX_NEW_TOKENS})
    scored: set[Path] = set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        generation = pool.submit(run_jobs, commands, devices, logs)
        while not generation.done():
            if sum(score_ready(key, scored) for key in MODELS):
                audit.main()
            try:
                generation.result(timeout=15)
            except TimeoutError:
                continue
        generation.result()
    for key in MODELS:
        score_ready(key, scored)
    if len(scored) != 36:
        raise RuntimeError(f"Only {len(scored)} of 36 cells finished scoring")
    audit.main()
    print("Finished all 36 A-LQR cells and refreshed the comparison heatmap.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--devices", default="auto")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("model", choices=MODELS)
    worker_parser.add_argument("device")
    worker_parser.add_argument("shard", type=int)
    worker_parser.add_argument("shards", type=int)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.model, args.device, args.shard, args.shards)
    else:
        OUTPUT.mkdir(exist_ok=True)
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(resolve_cuda_devices(args.devices))


if __name__ == "__main__":
    main()
