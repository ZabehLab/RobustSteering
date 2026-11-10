"""HarmBench inputs owned by the rebuttal unit."""

from __future__ import annotations

from dataclasses import asdict
import errno
import json
import os
from pathlib import Path
import shutil

import torch

from robust_steerability.benchmarks import harmful_artifacts as paper_fit
from robust_steerability.benchmarks.artifacts import aggregate_jacobian_partials
from robust_steerability.benchmarks.execution import utc_now
from robust_steerability.benchmarks.layout import REPO_ROOT
from robust_steerability.benchmarks.specs import MODELS, ModelSpec
from robust_steerability.calibration.nominal import average_prompt_jacobians
from robust_steerability.calibration.nominal_artifact import (
    nominal_dynamics_identity,
    nominal_dynamics_signature,
    save_nominal_dynamics,
)
from robust_steerability.experiments.calibration import calibrate_controller, diagnostic_root
from robust_steerability.modeling.huggingface import (
    CausalModelLoadSpec,
    load_access_token,
    load_causal_model,
    release_cuda_memory,
)
from robust_steerability.source_methods.control import fit_setpoint_calibration

UNIT = Path(__file__).resolve().parent
PAPER_CACHE = REPO_ROOT / "benchmarks/harmful/cache"
REFERENCE_MODEL = "llama32_3b_instruct"
SMALL_MODELS = {
    "smollm2_360m_instruct": ModelSpec(
        "smollm2_360m_instruct", "SmolLM2-360M-Instruct",
        "HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951", 16, 32,
    ),
    "qwen25_05b_instruct": ModelSpec(
        "qwen25_05b_instruct", "Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", 16, 32,
    ),
}
MODEL_SPECS = {
    **{key: MODELS[key] for key in paper_fit.MODEL_KEYS if MODELS[key].devices_per_worker == 1},
    **SMALL_MODELS,
}
COORDINATES = ("means", "encoders", "decoders", "feature_unit", "setpoints")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def write_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_tensors(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_spec(spec: ModelSpec) -> CausalModelLoadSpec:
    return CausalModelLoadSpec(spec.model_id, revision=spec.revision, attention_implementation="sdpa")


def load_model(spec: ModelSpec, device: str):
    return load_causal_model(load_spec(spec), device, load_access_token(REPO_ROOT))


def archive_paths(model_key: str) -> dict[str, Path]:
    root = PAPER_CACHE / model_key
    return {
        "data": root / "artifacts/data.json",
        "evaluation": root / "datasets/harmbench.json",
        "setpoint": root / "artifacts/setpoint.pt",
        "dynamics": root / "artifacts/dynamics.pt",
        "base": root / "calibrations/h_infinity/selected/base/controller.pt",
    }


def reduced_inputs(base_path: Path, dynamics_path: Path, spec: ModelSpec) -> dict:
    base = load_tensors(base_path)
    bundle = load_tensors(diagnostic_root(base_path) / "input.pt")
    settings = bundle["calibration"]["settings"]
    identity = nominal_dynamics_signature(dynamics_path)
    nominal = identity["identity"]
    if (nominal.get("model_id"), nominal.get("model_revision")) != (spec.model_id, spec.revision):
        raise ValueError("The nominal dynamics checkpoint does not match the run model")
    if base["metadata"]["nominal_dynamics"] != identity:
        raise ValueError("The reduced and full-width controllers must share the same nominal dynamics")
    if settings["state_rank"] != 8 or settings["hinf_setpoint_multiplier"] != 1.5:
        raise ValueError("Expected the frozen rank-eight HarmBench fit and setpoint multiplier 1.5")
    return {
        "fingerprint": base["fingerprint"],
        "problem": bundle["problem"],
        "options": bundle["options"],
        "settings": settings,
        "coordinates": {name: base["artifact"][name] for name in COORDINATES},
    }


def prepare_records(root: Path, spec: ModelSpec, tokenizer) -> dict:
    """Retokenize the exact archived paper inputs for a new checkpoint."""
    reference = archive_paths(REFERENCE_MODEL)
    data = read_json(reference["data"])
    evaluation = read_json(reference["evaluation"])
    calibration = data["calibration"]
    data["model"] = [spec.model_id, spec.revision]
    for label in ("desired", "undesired"):
        calibration[label] = [paper_fit._direction_record(tokenizer, row) for row in calibration[label]]
    calibration["jacobian"] = calibration["desired"][:paper_fit.JACOBIAN_PROMPTS]
    calibration["tuning"] = [
        {**row, "text": paper_fit._chat_prompt(tokenizer, row["raw_request"])}
        for row in calibration["tuning"]
    ]
    calibration["disturbance"] = calibration["tuning"]
    for rows in evaluation["evaluation"].values():
        for row in rows:
            row["model_input"] = paper_fit._chat_prompt(tokenizer, row["text"])
    write_json(root / "data.json", data)
    write_json(root / "harmbench.json", evaluation)
    return data


def jacobian_worker(root: Path, spec: ModelSpec, device: str, index: int, count: int) -> None:
    model, tokenizer = load_model(spec, device)
    records = read_json(root / "data.json")["calibration"]["jacobian"][index::count]
    average_prompt_jacobians(
        model, tokenizer, records, cache_dir=root / f"jacobians/{count:02d}-{index:02d}",
        max_length=paper_fit.JACOBIAN_MAX_LENGTH,
        vjp_chunk_size=spec.jacobian_vjp_chunk_size, model_revision=spec.revision,
    )


def fit_fresh(root: Path, spec: ModelSpec, devices: list[str], launch_jacobians) -> dict[str, Path]:
    """Fit a fresh checkpoint inside the unit with the paper's inputs and solver."""
    paths = {
        "data": root / "data.json",
        "evaluation": root / "harmbench.json",
        "setpoint": root / "setpoint.pt",
        "dynamics": root / "dynamics.pt",
        "base": root / "base/controller.pt",
    }
    if not paths["setpoint"].exists():
        model, tokenizer = load_model(spec, devices[0])
        data = prepare_records(root, spec, tokenizer)
        calibration = data["calibration"]
        means = [
            paper_fit._class_response_mean(model, tokenizer, calibration[label], spec.activation_batch_size)
            for label in ("undesired", "desired")
        ]
        setpoint = fit_setpoint_calibration(*means)
        write_torch(paths["setpoint"], asdict(setpoint))
        del model, tokenizer
        release_cuda_memory(devices[0])
    data = read_json(paths["data"])
    calibration = data["calibration"]
    records = calibration["jacobian"]
    if not paths["dynamics"].exists() or not paths["dynamics"].with_suffix(".json").exists():
        worker_devices = devices[:len(records)]
        launch_jacobians(worker_devices)
        directories = [root / f"jacobians/{len(worker_devices):02d}-{i:02d}" for i in range(len(worker_devices))]
        counts = [len(records[i::len(worker_devices)]) for i in range(len(worker_devices))]
        identity = nominal_dynamics_identity(
            behavior=paper_fit.CONCEPT, model_id=spec.model_id, model_revision=spec.revision,
            records=records, max_length=paper_fit.JACOBIAN_MAX_LENGTH,
            vjp_chunk_size=spec.jacobian_vjp_chunk_size,
        )
        save_nominal_dynamics(
            paths["dynamics"], identity, aggregate_jacobian_partials(directories, counts),
            attempts=[{"status": "complete", "created_at_utc": utc_now()}],
        )

    if not paths["base"].exists():
        reference = archive_paths(REFERENCE_MODEL)
        reference_bundle = load_tensors(diagnostic_root(reference["base"]) / "input.pt")
        settings = dict(reference_bundle["calibration"]["settings"])
        settings.update(
            activation_batch_size=spec.activation_batch_size,
            jacobian_vjp_chunk_size=spec.jacobian_vjp_chunk_size,
            model_loading=asdict(load_spec(spec)),
        )
        model, tokenizer = load_model(spec, devices[0])
        calibrate_controller(
            model, tokenizer, model_label=spec.label, model_id=spec.model_id,
            cache_path=paths["base"], nominal_dynamics_path=paths["dynamics"],
            calibration_data={
                "negative": calibration["undesired"], "positive": calibration["desired"],
                "jacobian": records, "disturbance": calibration["disturbance"],
                "dataset": {"direction": "archived HarmBench human-majority completions",
                            "disturbance_and_tuning": "50 direct validation behaviors",
                            "evaluation": "240 held-out test behaviors"},
            },
            settings=settings, controller_device=devices[0],
            semantic_calibration=load_tensors(paths["setpoint"]),
        )
        del model, tokenizer
        release_cuda_memory(devices[0])
    return paths


def prepare_inputs(root: Path, spec: ModelSpec, devices: list[str], launch_jacobians) -> dict:
    manifest = root / "manifest.json"
    if manifest.exists():
        saved = read_json(manifest)
        if saved["model"] != [spec.model_id, spec.revision]:
            raise ValueError("This model folder belongs to a different checkpoint")
        return saved

    archived = archive_paths(spec.key)
    reuse = all(path.exists() for path in archived.values()) and (
        diagnostic_root(archived["base"]) / "input.pt"
    ).exists()
    paths = archived if reuse else fit_fresh(root, spec, devices, launch_jacobians)
    data = read_json(paths["data"])
    if data["model"] != [spec.model_id, spec.revision]:
        raise ValueError("The HarmBench calibration records are tokenized for a different model")
    fit = reduced_inputs(paths["base"], paths["dynamics"], spec)
    write_torch(root / "reduced.pt", fit)
    write_json(root / "data.json", data)
    write_json(root / "harmbench.json", read_json(paths["evaluation"]))
    if reuse:
        setpoint = load_tensors(paths["setpoint"])
        write_torch(root / "setpoint.pt", {name: setpoint[name] for name in ("contrast", "feature_norm")})
        dynamics = root / "dynamics.pt"
        if not dynamics.exists():
            temporary = dynamics.with_suffix(".pt.tmp")
            temporary.unlink(missing_ok=True)
            try:
                os.link(paths["dynamics"], temporary)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copyfile(paths["dynamics"], temporary)
            temporary.replace(dynamics)
        write_json(dynamics.with_suffix(".json"), read_json(paths["dynamics"].with_suffix(".json")))
    else:
        dynamics = paths["dynamics"]
    saved = {
        "model": [spec.model_id, spec.revision],
        "source": "paper_fit" if reuse else "unit_fit",
        "source_paths": {name: str(path) for name, path in paths.items()},
        "nominal_dynamics": nominal_dynamics_signature(dynamics),
    }
    write_json(manifest, saved)
    return saved
