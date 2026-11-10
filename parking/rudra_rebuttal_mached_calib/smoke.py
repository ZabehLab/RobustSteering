"""Exercise the real Llama 3B generation hooks on a neutral prompt."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from robust_steerability.benchmarks.execution import default_run_id
from robust_steerability.benchmarks.harmful_calibration import grid
from robust_steerability.benchmarks.harmful_runtime import generate_completions
from robust_steerability.benchmarks.launcher import run_jobs
from robust_steerability.experiments.resources import resolve_cuda_devices
from robust_steerability.source_methods.control import SetpointCalibration

from .inputs import MODEL_SPECS, REFERENCE_MODEL, load_model, load_tensors, prepare_inputs, read_json, write_json
from .pipeline import CACHE
from .workers import METHODS, candidate_policy, synthesize_candidate


def worker(root: Path, methods: list[str], device: str, output: Path) -> None:
    torch.cuda.set_device(device)
    model, tokenizer = load_model(MODEL_SPECS[REFERENCE_MODEL], device)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name a color in one word."}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = root / "artifacts"
    fit = load_tensors(inputs / "reduced.pt")
    setpoint = SetpointCalibration(**load_tensors(inputs / "setpoint.pt"))
    config = next(item for item in grid() if item["grid_id"] == "q_01_qf_00")
    results = {}
    for method in methods:
        dynamics = None
        if method == "alqr":
            dynamics = load_tensors(inputs / "dynamics.pt")["dynamics"]
        candidate = synthesize_candidate(method, fit, config, setpoint, dynamics, device)
        if candidate["status"] != "complete":
            raise RuntimeError(f"{method} failed to synthesize")
        candidate["identity"] = {"configuration": config}
        policy = candidate_policy(method, fit, candidate, setpoint, record=True)
        _, (tokens,) = generate_completions(
            model, tokenizer, [prompt], steering_policy=policy, use_cache=False,
            batch_size=1, max_new_tokens=2,
        )
        trace = policy.recorder.finish()
        energy = trace["hidden_delta_energy_total"]
        if not torch.isfinite(trace["control_energy_step"]).all() or not energy > 0:
            raise RuntimeError(f"{method} failed to apply a finite nonzero intervention")
        if trace["layer_index"].unique().numel() != len(candidate["gains"]):
            raise RuntimeError(f"{method} did not intervene on every fitted decoder layer")
        results[method] = {"device": device, "generated_tokens": tokens,
                           "hook_calls": len(trace["layer_index"]), "hidden_delta_energy": energy}
        print(method, results[method], flush=True)
        del candidate, policy, dynamics
    write_json(output, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-methods", nargs="+", choices=METHODS)
    parser.add_argument("--device")
    args = parser.parse_args()
    root = CACHE / REFERENCE_MODEL
    if args.worker_output:
        worker(root, args.worker_methods, args.device, args.worker_output)
        return
    devices = resolve_cuda_devices(args.devices)

    def cached_fit_required(_devices):
        raise RuntimeError("The smoke check requires the archived 3B fit")

    prepare_inputs(root / "artifacts", MODEL_SPECS[REFERENCE_MODEL], devices, cached_fit_required)
    logs = root / "checks" / default_run_id("matched-calib", REFERENCE_MODEL, "smoke")
    count = min(len(devices), len(METHODS))
    jobs = [
        (
            f"device-{index}",
            [sys.executable, "-m", "parking.rudra_rebuttal_mached_calib.smoke",
             "--worker-output", str(logs / f"device-{index}.json"),
             "--worker-methods", *METHODS[index::count], "--device", "{device}"],
        )
        for index in range(count)
    ]
    run_jobs(jobs, devices[:count], logs)
    merged = {}
    for index in range(count):
        merged.update(read_json(logs / f"device-{index}.json"))
    if set(merged) != set(METHODS):
        raise RuntimeError("Missing method smoke results")
    write_json(logs / "results.json", merged)
    print(logs / "results.json")


if __name__ == "__main__":
    main()
