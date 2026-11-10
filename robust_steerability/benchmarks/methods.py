"""Shared method registry for benchmark orchestration.

Benchmark drivers select method keys from their TOML composition.  Method-specific
math remains in the owning implementation; this registry contains only the small
amount of orchestration metadata needed by every benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    runtime: str
    calibration: str
    model_loader: str
    policy_key: str | None = None

    @property
    def requires_source_fit(self) -> bool:
        return self.calibration == "source_fit"


METHODS = {
    "original": MethodSpec("original", "Original", "source", "none", "alqr"),
    "iti": MethodSpec("iti", "ITI", "source", "source_fit", "iti"),
    "actadd": MethodSpec("actadd", "ActAdd", "source", "source_fit", "actadd"),
    "mean_act": MethodSpec(
        "mean_act", "Mean-AcT", "source", "source_fit", "mean_act"
    ),
    "linear_act": MethodSpec(
        "linear_act", "Linear-AcT", "source", "source_fit", "linear_act"
    ),
    "pid_act": MethodSpec(
        "pid_act", "PID-AcT", "source", "source_fit", "pid_act"
    ),
    "odesteer": MethodSpec(
        "odesteer", "ODESteer", "source", "source_fit", "odesteer"
    ),
    "spid": MethodSpec("spid", "S-PID", "source", "source_fit", "alqr"),
    "alqr": MethodSpec("alqr", "A-LQR", "source", "fixed", "alqr"),
    "alqr_8d": MethodSpec("alqr_8d", "A-LQR (8D)", "alqr_8d", "alqr_8d", "alqr"),
    "h_infinity": MethodSpec(
        "h_infinity", "H-infinity", "h_infinity", "h_infinity", "alqr", "hinf"
    ),
}

METHOD_KEYS = tuple(METHODS)


def method_spec(key: str) -> MethodSpec:
    if key not in METHODS:
        raise ValueError(f"Unknown benchmark method {key!r}")
    return METHODS[key]


def validate_methods(methods: tuple[str, ...] | list[str]) -> None:
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown benchmark methods: {sorted(unknown)}")
