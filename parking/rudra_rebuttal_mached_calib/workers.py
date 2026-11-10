"""Independent generation and scoring workers for the rebuttal unit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

from robust_steerability.artifacts import configuration_hash
from robust_steerability.benchmarks.harmful_calibration import DEFAULT_GENERATION_BATCH_SIZE
from robust_steerability.benchmarks.harmful_runtime import (
    DEFAULT_BATCH_SIZE, MAX_NEW_TOKENS, evaluation_records, generate_completions,
)
from robust_steerability.benchmarks.layout import REPO_ROOT
from robust_steerability.control import FiniteHorizonControlProblem, HInfinityController, LQRController
from robust_steerability.control.h_infinity import HInfinityOptions
from robust_steerability.datasets.harmful import SAFE_CONCEPT
from robust_steerability.experiments.methods import ReducedLQRArtifact, build_policy
from robust_steerability.judges.harmbench import score_generations
from robust_steerability.modeling.huggingface import load_access_token
from robust_steerability.runtime.diagnostics import ReducedTrajectoryRecorder
from robust_steerability.runtime.policy import ReducedStateSetpointPolicy, SemanticSetpointPolicy
from robust_steerability.source_methods.control import SetpointCalibration, build_alqr_policy

from .inputs import MODEL_SPECS, SMALL_MODELS, load_model, load_tensors, read_json, write_json, write_torch

METHODS = ("alqr", "alqr_8d", "h_infinity")
GIGA_METHODS = ("original", *METHODS)
CONDITIONS = ("direct", "human_jailbreak")
CALIBRATION_PROMPTS = {"mini": 20, "mega": 50, "full": 50}
PROFILES = (*CALIBRATION_PROMPTS, "giga")
MINI_BEHAVIORS = 20
EVAL_SHARDS = 2


def candidate_problem(fit: dict, config: dict) -> FiniteHorizonControlProblem:
    problem = FiniteHorizonControlProblem(**fit["problem"])
    base = fit["settings"]
    return replace(
        problem,
        state_costs=problem.state_costs * (config["q"] / base["q"]),
        control_costs=problem.control_costs * (config["r"] / base["r"]),
        terminal_cost=problem.terminal_cost * (config["q_final"] / base["q_final"]),
    )


def synthesize_candidate(method: str, fit: dict, config: dict, setpoint: SetpointCalibration,
                         dynamics: torch.Tensor | None, device: str) -> dict:
    if method == "alqr":
        if dynamics is None:
            raise ValueError("Full-width A-LQR requires the shared nominal dynamics")
        policy = build_alqr_policy(
            dynamics, setpoint, multiplier=config["lambda"], q=config["q"],
            r=config["r"], q_final=config["q_final"], device=device,
        )
        solution = policy.controller.solution()
    else:
        problem = candidate_problem(fit, config)
        if method == "alqr_8d":
            solution = LQRController.synthesize(replace(problem, disturbance_channels=None), device=device).solution()
        elif method == "h_infinity":
            solution = HInfinityController.synthesize(
                problem, device=device, options=HInfinityOptions(**fit["options"])
            ).solution()
        else:
            raise ValueError(f"Unknown controller: {method}")
    return {
        "status": "complete" if solution.feasible else "infeasible",
        "gains": solution.gains, "gamma_star": solution.gamma_star,
        "diagnostics": solution.diagnostics,
    }


def candidate_policy(method: str, fit: dict, candidate: dict, setpoint: SetpointCalibration,
                     *, record: bool):
    gains = candidate["gains"]
    if method == "alqr":
        policy = SemanticSetpointPolicy(
            LQRController(gains), setpoint.unit_features(0.0),
            setpoint.setpoints(candidate["identity"]["configuration"]["lambda"]),
        )
    elif method == "alqr_8d":
        policy = build_policy(
            method, ReducedLQRArtifact(
                gains=gains, control_channels=fit["problem"]["control_channels"], **fit["coordinates"]
            ), kp=0, ki=0, kd=0,
        )
    elif method == "h_infinity":
        policy = ReducedStateSetpointPolicy(
            HInfinityController(
                gains=gains, control_channels=fit["problem"]["control_channels"], feasible=True,
                gamma_star=candidate["gamma_star"], diagnostics=candidate["diagnostics"],
            ), **fit["coordinates"],
        )
    else:
        raise ValueError(f"Unknown controller: {method}")
    if record:
        policy.recorder = ReducedTrajectoryRecorder("norms")
    return policy


def calibration_records(tuning: list[dict], profile: str) -> list[dict]:
    count = CALIBRATION_PROMPTS[profile]
    if len(tuning) < count:
        raise ValueError(f"Requested {count} calibration prompts from {len(tuning)} available")
    return tuning[:count]


def evaluation_split(source: dict, condition: str, part: str) -> list[dict]:
    mini = evaluation_records(source, condition, MINI_BEHAVIORS)
    if part == "screening":
        return mini
    if part != "holdout":
        raise ValueError(f"Unknown evaluation part: {part}")
    mini_ids = {row["prompt_id"] for row in mini}
    return [row for row in evaluation_records(source, condition, None) if row["prompt_id"] not in mini_ids]


def candidate_path(root: Path, task: dict) -> Path:
    return root / "calibration/controllers" / task["method"] / f"{task['configuration']['grid_id']}.pt"


def generation_path(root: Path, task: dict) -> Path:
    stage, method = task["stage"], task["method"]
    directory = generation_root(root, stage, task["profile"]) / "generations"
    if stage == "calibration":
        return directory / method / f"{task['configuration']['grid_id']}.json"
    return directory / task["condition"] / method / f"shard-{task['shard']}.json"


def generation_root(root: Path, stage: str, profile: str) -> Path:
    if stage == "calibration":
        return root / "calibration" / profile
    return root / "evaluation" / profile / stage


def task_inputs(model_key: str, task: dict, tuning: list[dict], evaluation: dict) -> tuple[list[dict], int]:
    small_model_screening = model_key in SMALL_MODELS and task["profile"] in {"mini", "mega", "giga"}
    if task["stage"] == "calibration":
        records = calibration_records(tuning, task["profile"])
        batch_size = CALIBRATION_PROMPTS["mini"] if small_model_screening else DEFAULT_GENERATION_BATCH_SIZE.get(model_key, 16)
        return records, batch_size
    condition = task["condition"]
    records = evaluation_split(evaluation, condition, task["stage"])
    batch_size = DEFAULT_BATCH_SIZE.get(model_key, {"direct": 16, "human_jailbreak": 8})[condition]
    if small_model_screening and condition == "human_jailbreak":
        batch_size = 25
    if task["profile"] == "giga" and task["stage"] == "holdout":
        # Nearby prompt lengths reduce padding when the entire prefix is processed each step.
        records = sorted(records, key=lambda row: (len(row["text"]), row["prompt_id"]))
        batch_size = min(batch_size, 16)
    return records[task["shard"]::EVAL_SHARDS], batch_size


def controller_identity(fit_fingerprint: str, model: list[str], task: dict) -> dict:
    return {"model": model, "fit": fit_fingerprint, "method": task["method"],
            "configuration": task["configuration"]}


def generation_identity(candidate_id: dict, task: dict, records: list[dict], batch_size: int) -> dict:
    return {
        **candidate_id, "profile": task["profile"], "stage": task["stage"],
        "condition": task.get("condition", "direct"),
        "prompt_hash": configuration_hash({"records": records}), "use_cache": False,
        "max_new_tokens": MAX_NEW_TOKENS, "batch_size": batch_size,
    }


def run_generation_task(root: Path, task: dict, records: list[dict], identity: dict,
                        model, tokenizer, policy, batch_size: int) -> None:
    path = generation_path(root, task)
    calibrating = task["stage"] == "calibration"
    payload = read_json(path) if path.exists() else {
        "identity": identity, "status": "partial", "repetitions": [{"repetition": 0, "rows": []}],
    }
    rows = payload["repetitions"][0]["rows"]
    if payload["identity"] != identity or [row["prompt_id"] for row in rows] != [
        row["prompt_id"] for row in records[:len(rows)]
    ]:
        raise ValueError(f"Cached generation does not match its frozen inputs: {path}")
    if payload["status"] == "complete":
        if len(rows) != len(records):
            raise ValueError(f"Incomplete cache marked complete: {path}")
        return

    while len(rows) < len(records):
        start = len(rows)
        end = len(records)
        if calibrating and task["profile"] == "mega" and start < CALIBRATION_PROMPTS["mini"]:
            end = CALIBRATION_PROMPTS["mini"]
        batch = records[start:min(start + batch_size, end)]
        started = perf_counter()
        completions, counts = generate_completions(
            model, tokenizer, [row["text" if calibrating else "model_input"] for row in batch],
            steering_policy=policy, use_cache=False, batch_size=len(batch), max_new_tokens=MAX_NEW_TOKENS,
        )
        energy = (
            policy.recorder.finish()["control_energy_step"].sum(dim=0).tolist()
            if calibrating else [None] * len(batch)
        )
        for row, completion, count, control_energy in zip(batch, completions, counts, energy, strict=True):
            scored = {
                key: row[key] for key in ("prompt_id", "behavior_id", "behavior", "context",
                                          "functional_category", "semantic_category")
            }
            scored.update(
                text=row["raw_request"] if calibrating else row["text"], concept=SAFE_CONCEPT,
                completion=completion, generated_tokens=count,
            )
            if calibrating:
                scored["control_energy"] = control_energy
                scored["grid_id"] = task["configuration"]["grid_id"]
            rows.append(scored)
        payload["repetitions"][0]["sample_count"] = len(rows)
        payload["status"] = "complete" if len(rows) == len(records) else "partial"
        write_json(path, payload)
        print(f"{task['stage']}/{task['method']}: saved {len(rows)}/{len(records)} responses "
              f"(batch {len(batch)}, {perf_counter() - started:.1f}s)", flush=True)


def generate_worker(root: Path, model_key: str, device: str, tasks: list[dict]) -> None:
    spec = MODEL_SPECS[model_key]
    model, tokenizer = load_model(spec, device)
    if model_key in SMALL_MODELS and model.generation_config.eos_token_id is not None:
        # The shared generator reads the model config, so honor the checkpoint's generation stops there.
        model.config.eos_token_id = model.generation_config.eos_token_id
    inputs = root / "artifacts"
    manifest = read_json(inputs / "manifest.json")
    fit = load_tensors(inputs / "reduced.pt")
    setpoint = SetpointCalibration(**load_tensors(inputs / "setpoint.pt"))
    tuning = read_json(inputs / "data.json")["calibration"]["tuning"]
    evaluation = read_json(inputs / "harmbench.json")
    dynamics = None
    for task in tasks:
        method, stage, config = task["method"], task["stage"], task["configuration"]
        calibrating = stage == "calibration"
        candidate_id = controller_identity(fit["fingerprint"], manifest["model"], task)
        records, batch_size = task_inputs(model_key, task, tuning, evaluation)
        identity = generation_identity(candidate_id, task, records, batch_size)
        if method == "original":
            run_generation_task(root, task, records, identity, model, tokenizer, None, batch_size)
            continue
        destination = candidate_path(root, task)
        if calibrating and not destination.exists():
            if method == "alqr" and dynamics is None:
                dynamics = load_tensors(inputs / "dynamics.pt")["dynamics"]
            try:
                candidate = synthesize_candidate(method, fit, config, setpoint, dynamics, device)
            except torch.linalg.LinAlgError as error:
                candidate = {"status": "infeasible", "error": str(error)}
            write_torch(destination, {"identity": candidate_id, **candidate})
        candidate = load_tensors(destination)
        if candidate["identity"] != candidate_id:
            raise ValueError(f"Controller identity mismatch: {destination}")
        if candidate["status"] != "complete":
            print(f"{method}/{config['grid_id']}: {candidate['status']}", flush=True)
            continue
        policy = candidate_policy(method, fit, candidate, setpoint, record=calibrating)
        run_generation_task(root, task, records, identity, model, tokenizer, policy, batch_size)
        print(f"{stage}/{method}/{config['grid_id']}: {len(records)} responses", flush=True)
        del policy, candidate


def classifier_worker(root: Path, device: str, stage: str, profile: str, paths: list[str]) -> None:
    scorer = "harmbench_validation_success" if stage == "calibration" else "harmbench_test_success"
    score_generations(
        [Path(path) for path in paths], generation_root(root, stage, profile), scorer, device,
        load_access_token(REPO_ROOT),
        batch_size=8 if stage == "calibration" else 1,
    )
