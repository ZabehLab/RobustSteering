"""Canonical controller names and reduced-state policy construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robust_steerability.benchmarks.methods import METHODS as BENCHMARK_METHODS
from robust_steerability.control import (
    HInfinityController,
    LQRController,
    PIDController,
    PIDGains,
)
from robust_steerability.runtime.policy import ReducedStateSetpointPolicy, SemanticSetpointPolicy
from robust_steerability.runtime.diagnostics import ReducedTrajectoryRecorder


METHOD_LABELS = {
    specification.policy_key or key: specification.label
    for key, specification in BENCHMARK_METHODS.items()
}


@dataclass
class ControllerArtifact:
    """Calibrated coordinates and controller solutions for one model."""

    means: torch.Tensor
    encoders: torch.Tensor
    decoders: torch.Tensor
    feature_unit: torch.Tensor
    setpoints: torch.Tensor
    control_channels: torch.Tensor
    lqr_gains: torch.Tensor
    hinf_gains: torch.Tensor
    hinf_feasible: bool
    gamma_star: float | None
    hinf_diagnostics: dict[str, object]
    baselines: dict
    raw_feature_unit: torch.Tensor
    alqr_setpoints: torch.Tensor
    spid_setpoints: torch.Tensor


@dataclass(frozen=True)
class ReducedLQRArtifact:
    """Nominal gains and the fitted eight-dimensional tracking coordinates."""

    gains: torch.Tensor
    control_channels: torch.Tensor
    means: torch.Tensor
    encoders: torch.Tensor
    decoders: torch.Tensor
    feature_unit: torch.Tensor
    setpoints: torch.Tensor

    def __post_init__(self) -> None:
        if self.gains.ndim != 3 or self.gains.shape[-1] != 8:
            raise ValueError("A-LQR (8D) requires gains on an eight-dimensional state")


def build_policy(
    method: str,
    artifact: ControllerArtifact | ReducedLQRArtifact,
    *,
    kp: float,
    ki: float,
    kd: float,
    record: bool = False,
):
    """Build the online policy for one canonical method name."""

    if method == "original":
        return None
    if method in {"iti", "actadd", "mean_act", "linear_act", "pid_act", "odesteer"}:
        from robust_steerability.experiments.baselines import BaselinePolicy
        return BaselinePolicy(method, artifact.baselines, strength=artifact.baselines["strengths"][method], record=record)
    if method == "alqr":
        controller = LQRController(artifact.lqr_gains)
    elif method == "alqr_8d":
        if not isinstance(artifact, ReducedLQRArtifact):
            raise TypeError("A-LQR (8D) requires a ReducedLQRArtifact")
        controller = LQRController(artifact.gains, artifact.control_channels)
    elif method == "spid":
        controller = PIDController(
            PIDGains(proportional=kp, integral=ki, derivative=kd),
        )
    elif method == "hinf":
        controller = HInfinityController(
            gains=artifact.hinf_gains,
            control_channels=artifact.control_channels,
            feasible=artifact.hinf_feasible,
            gamma_star=artifact.gamma_star,
            diagnostics=artifact.hinf_diagnostics,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    if method in {"alqr", "spid"}:
        setpoints = artifact.alqr_setpoints if method == "alqr" else artifact.spid_setpoints
        return SemanticSetpointPolicy(controller, artifact.raw_feature_unit, setpoints,
                                      recorder=ReducedTrajectoryRecorder("norms") if record else None)
    return ReducedStateSetpointPolicy(
        controller=controller,
        means=artifact.means,
        encoders=artifact.encoders,
        decoders=artifact.decoders,
        feature_unit=artifact.feature_unit,
        setpoints=artifact.setpoints,
        recorder=ReducedTrajectoryRecorder() if record else None,
    )
