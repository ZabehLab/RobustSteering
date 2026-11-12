"""Optimized finite-horizon H-infinity synthesis and online execution.

The recursion and gamma search follow the reference implementation preserved in
``ref/h_infinity.py``. Synthesis prepares problem tensors once and keeps
intermediate search evaluations lean; only the selected attenuation level
collects gains and the complete diagnostic surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from robust_steerability.control.base import SynthesizedController
from robust_steerability.control.types import (
    ControllerSolution,
    FiniteHorizonControlProblem,
)
from robust_steerability.control.validation import (
    validate_control_problem,
    validate_controller_solution,
)


@dataclass(frozen=True)
class HInfinityOptions:
    """Numerical search settings for finite-horizon H-infinity synthesis."""

    gamma_lower: float = 0.0
    gamma_upper: float = 1000.0
    tolerance: float = 1e-5
    max_iterations: int = 100
    numerical_tolerance: float = 1e-7
    max_gamma: float = 1e12
    deployment_margin: float = 0.0


@dataclass(frozen=True)
class _PreparedProblem:
    dynamics: torch.Tensor
    control_channels: torch.Tensor
    disturbance_channels: torch.Tensor
    state_costs: torch.Tensor
    control_costs: torch.Tensor
    terminal_cost: torch.Tensor
    disturbance_identity: torch.Tensor


@dataclass(frozen=True)
class _GammaSolve:
    feasible: bool
    gains: torch.Tensor | None
    diagnostics: dict[str, object] | None


def _empty_diagnostics() -> dict[str, object]:
    """Return the complete diagnostic surface for one gamma solve."""

    return {
        "min_disturbance_margin": float("inf"),
        "min_control_margin": float("inf"),
        "max_condition_number_M": 0.0,
        "max_condition_number_H": 0.0,
        "max_norm_S": 0.0,
        "max_norm_K": 0.0,
        "failed_layer": None,
        "failure_reason": None,
    }


def _prepare_problem(
    problem: FiniteHorizonControlProblem,
    device: str | torch.device,
) -> _PreparedProblem:
    disturbance_channels = problem.disturbance_channels
    assert disturbance_channels is not None
    dtype = torch.float32
    prepared_disturbances = disturbance_channels.to(device=device, dtype=dtype)
    return _PreparedProblem(
        dynamics=problem.dynamics.to(device=device, dtype=dtype),
        control_channels=problem.control_channels.to(device=device, dtype=dtype),
        disturbance_channels=prepared_disturbances,
        state_costs=problem.state_costs.to(device=device, dtype=dtype),
        control_costs=problem.control_costs.to(device=device, dtype=dtype),
        terminal_cost=problem.terminal_cost.to(device=device, dtype=dtype),
        disturbance_identity=torch.eye(
            prepared_disturbances.shape[2],
            device=device,
            dtype=dtype,
        ),
    )


def _solve_prepared_gamma(
    problem: _PreparedProblem,
    gamma: float,
    *,
    numerical_tolerance: float,
    collect: bool,
) -> _GammaSolve:
    """Run Hannah's backward game recursion on already prepared tensors."""

    horizon, state_dimension, _ = problem.dynamics.shape
    control_dimension = problem.control_channels.shape[2]
    value = problem.terminal_cost
    gains = (
        torch.zeros(
            horizon,
            control_dimension,
            state_dimension,
            device=value.device,
            dtype=value.dtype,
        )
        if collect
        else None
    )
    diagnostics = _empty_diagnostics() if collect else None
    if diagnostics is not None:
        diagnostics["max_norm_S"] = float(torch.linalg.matrix_norm(value).item())

    for layer_index in reversed(range(horizon)):
        layer_dynamics = problem.dynamics[layer_index]
        control_channel = problem.control_channels[layer_index]
        disturbance_channel = problem.disturbance_channels[layer_index]

        value_times_disturbance = value @ disturbance_channel
        disturbance_hessian = (
            gamma**2 * problem.disturbance_identity
            - disturbance_channel.T @ value_times_disturbance
        )
        disturbance_hessian = 0.5 * (
            disturbance_hessian + disturbance_hessian.T
        )
        disturbance_eigenvalues = torch.linalg.eigvalsh(disturbance_hessian)
        disturbance_margin = float(disturbance_eigenvalues[0].item())
        if diagnostics is not None:
            diagnostics["min_disturbance_margin"] = min(
                float(diagnostics["min_disturbance_margin"]),
                disturbance_margin,
            )
        if disturbance_margin <= numerical_tolerance:
            if diagnostics is not None:
                diagnostics["failed_layer"] = layer_index
                diagnostics["failure_reason"] = (
                    "disturbance Hessian is not positive definite"
                )
            return _GammaSolve(False, gains, diagnostics)

        if diagnostics is not None:
            diagnostics["max_condition_number_M"] = max(
                float(diagnostics["max_condition_number_M"]),
                float((disturbance_eigenvalues[-1] / disturbance_eigenvalues[0]).item()),
            )
        disturbance_response = torch.linalg.solve(
            disturbance_hessian,
            value_times_disturbance.T,
        )
        robust_value = value + value_times_disturbance @ disturbance_response
        robust_value = 0.5 * (robust_value + robust_value.T)

        control_hessian = (
            problem.control_costs[layer_index]
            + control_channel.T @ robust_value @ control_channel
        )
        control_hessian = 0.5 * (control_hessian + control_hessian.T)
        control_eigenvalues = torch.linalg.eigvalsh(control_hessian)
        control_margin = float(control_eigenvalues[0].item())
        if diagnostics is not None:
            diagnostics["min_control_margin"] = min(
                float(diagnostics["min_control_margin"]),
                control_margin,
            )
        if control_margin <= numerical_tolerance:
            if diagnostics is not None:
                diagnostics["failed_layer"] = layer_index
                diagnostics["failure_reason"] = (
                    "control Hessian is not positive definite"
                )
            return _GammaSolve(False, gains, diagnostics)

        if diagnostics is not None:
            diagnostics["max_condition_number_H"] = max(
                float(diagnostics["max_condition_number_H"]),
                float((control_eigenvalues[-1] / control_eigenvalues[0]).item()),
            )
        cross_term = control_channel.T @ robust_value @ layer_dynamics
        positive_gain = torch.linalg.solve(control_hessian, cross_term)
        if gains is not None:
            gains[layer_index] = -positive_gain
        value = (
            problem.state_costs[layer_index]
            + layer_dynamics.T @ robust_value @ layer_dynamics
            - cross_term.T @ positive_gain
        )
        value = 0.5 * (value + value.T)
        if diagnostics is not None:
            diagnostics["max_norm_S"] = max(
                float(diagnostics["max_norm_S"]),
                float(torch.linalg.matrix_norm(value).item()),
            )
            diagnostics["max_norm_K"] = max(
                float(diagnostics["max_norm_K"]),
                float(torch.linalg.matrix_norm(positive_gain).item()),
            )

    return _GammaSolve(True, gains, diagnostics)


class HInfinityController(SynthesizedController):
    """Finite-horizon H-infinity controller.

    Offline synthesis computes the layer gains and attenuation boundary.
    Online, :meth:`control` computes ``u[k] = K[k] x[k]`` and the inherited
    :meth:`intervention` method returns ``B[k] u[k]``.
    """

    def __init__(
        self,
        gains: torch.Tensor,
        control_channels: torch.Tensor,
        *,
        feasible: bool,
        gamma_star: float | None,
        diagnostics: dict[str, object] | None = None,
    ):
        super().__init__(control_channels)
        if gains.ndim != 3:
            raise ValueError("gains must have shape (horizon, control, state)")
        if (
            control_channels.ndim != 3
            or control_channels.shape[0] != gains.shape[0]
            or control_channels.shape[1] != gains.shape[2]
            or control_channels.shape[2] != gains.shape[1]
        ):
            raise ValueError(
                "control_channels must have shape (horizon, state, control)"
            )
        if not torch.isfinite(gains).all():
            raise ValueError("gains contain non-finite values")
        self.register_buffer("gains", gains)
        self.feasible = feasible
        self.gamma_star = gamma_star
        self.diagnostics = diagnostics or {}

    @classmethod
    def synthesize_at_gamma(
        cls,
        problem: FiniteHorizonControlProblem,
        gamma: float,
        *,
        gamma_star: float | None = None,
        device: str | torch.device | None = None,
        numerical_tolerance: float = 1e-7,
    ) -> HInfinityController:
        """Solve at a specified attenuation level without searching for it.

        ``gamma_star`` optionally records an independently computed feasibility
        boundary; this method does not estimate that boundary or apply a margin.
        An infeasible solve returns a controller whose online use is prohibited.
        """

        validate_control_problem(problem)
        if problem.disturbance_channels is None:
            raise ValueError("H-infinity synthesis requires disturbance_channels")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        if not math.isfinite(numerical_tolerance) or numerical_tolerance <= 0:
            raise ValueError("numerical_tolerance must be finite and positive")
        if gamma_star is not None and (not math.isfinite(gamma_star) or gamma_star <= 0):
            raise ValueError("gamma_star must be finite and positive when supplied")
        target_device = device if device is not None else problem.dynamics.device
        with torch.inference_mode():
            result = _solve_prepared_gamma(
                _prepare_problem(problem, target_device), gamma,
                numerical_tolerance=numerical_tolerance, collect=True,
            )
        assert result.gains is not None and result.diagnostics is not None
        diagnostics = {
            **result.diagnostics,
            "synthesis": "fixed_gamma",
            "gamma_star": gamma_star,
            "gamma_used": gamma,
            "feedback_convention": "u[k] = K[k] x[k]",
        }
        solution = ControllerSolution(
            controller="h_infinity", gains=result.gains.detach().cpu(),
            feasible=result.feasible, gamma_star=gamma_star, diagnostics=diagnostics,
        )
        return cls.from_solution(problem, solution).to(target_device)

    @classmethod
    def synthesize(
        cls,
        problem: FiniteHorizonControlProblem,
        *,
        device: str | torch.device | None = None,
        options: HInfinityOptions = HInfinityOptions(),
    ) -> HInfinityController:
        """Synthesize the smallest feasible finite-horizon attenuation level."""

        validate_control_problem(problem)
        if problem.disturbance_channels is None:
            raise ValueError("H-infinity synthesis requires disturbance_channels")
        if options.gamma_lower < 0.0:
            raise ValueError("gamma_lower must be non-negative")
        if options.gamma_upper <= options.gamma_lower:
            raise ValueError("gamma_upper must be greater than gamma_lower")
        if options.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if options.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if options.numerical_tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be positive")
        if options.max_gamma < options.gamma_upper:
            raise ValueError("max_gamma must be at least gamma_upper")
        if options.deployment_margin < 0.0:
            raise ValueError("deployment_margin must be non-negative")

        target_device = device if device is not None else problem.dynamics.device
        with torch.inference_mode():
            prepared = _prepare_problem(problem, target_device)

            lower = options.gamma_lower
            lower_result = _solve_prepared_gamma(
                prepared,
                lower,
                numerical_tolerance=options.numerical_tolerance,
                collect=False,
            )
            while lower_result.feasible and lower > 0.0:
                lower *= 0.5
                lower_result = _solve_prepared_gamma(
                    prepared,
                    lower,
                    numerical_tolerance=options.numerical_tolerance,
                    collect=False,
                )

            upper = options.gamma_upper
            upper_result = _solve_prepared_gamma(
                prepared,
                upper,
                numerical_tolerance=options.numerical_tolerance,
                collect=False,
            )
            upper_bound_expansions = 0
            while not upper_result.feasible and upper < options.max_gamma:
                upper = min(2.0 * upper, options.max_gamma)
                upper_result = _solve_prepared_gamma(
                    prepared,
                    upper,
                    numerical_tolerance=options.numerical_tolerance,
                    collect=False,
                )
                upper_bound_expansions += 1

            if not upper_result.feasible:
                final_result = _solve_prepared_gamma(
                    prepared,
                    upper,
                    numerical_tolerance=options.numerical_tolerance,
                    collect=True,
                )
                assert final_result.gains is not None
                assert final_result.diagnostics is not None
                diagnostics = dict(final_result.diagnostics)
                diagnostics.update(
                    {
                        "gamma_star": None,
                        "gamma_used": None,
                        "gamma_lower_final": lower,
                        "gamma_upper_final": upper,
                        "bisection_iterations": 0,
                        "bisection_converged": False,
                        "upper_bound_expansions": upper_bound_expansions,
                        "feedback_convention": "u[k] = K[k] x[k]",
                    }
                )
                return cls(
                    gains=final_result.gains.detach().cpu(),
                    control_channels=problem.control_channels.detach().cpu(),
                    feasible=False,
                    gamma_star=None,
                    diagnostics=diagnostics,
                ).to(target_device)

            bisection_iterations = 0
            while (
                upper - lower >= options.tolerance
                and bisection_iterations < options.max_iterations
            ):
                midpoint = 0.5 * (lower + upper)
                midpoint_result = _solve_prepared_gamma(
                    prepared,
                    midpoint,
                    numerical_tolerance=options.numerical_tolerance,
                    collect=False,
                )
                if midpoint_result.feasible:
                    upper = midpoint
                else:
                    lower = midpoint
                bisection_iterations += 1

            gamma_star = upper
            gamma_used = (1.0 + options.deployment_margin) * gamma_star
            final_result = _solve_prepared_gamma(
                prepared,
                gamma_used,
                numerical_tolerance=options.numerical_tolerance,
                collect=True,
            )
            assert final_result.gains is not None
            assert final_result.diagnostics is not None
            diagnostics = dict(final_result.diagnostics)
            diagnostics.update(
                {
                    "gamma_star": gamma_star,
                    "gamma_used": gamma_used,
                    "robust_steerability": 1.0 / gamma_star,
                    "gamma_lower_final": lower,
                    "gamma_upper_final": upper,
                    "bisection_iterations": bisection_iterations,
                    "bisection_converged": upper - lower < options.tolerance,
                    "upper_bound_expansions": upper_bound_expansions,
                    "feedback_convention": "u[k] = K[k] x[k]",
                }
            )
            solution = ControllerSolution(
                controller="h_infinity",
                gains=final_result.gains.detach().cpu(),
                feasible=final_result.feasible,
                gamma_star=gamma_star,
                diagnostics=diagnostics,
            )
            validate_controller_solution(problem, solution)
            return cls.from_solution(problem, solution).to(target_device)

    @classmethod
    def from_solution(
        cls,
        problem: FiniteHorizonControlProblem,
        solution: ControllerSolution,
    ) -> HInfinityController:
        """Construct online H-infinity behavior from synthesized outputs."""

        validate_control_problem(problem)
        if problem.disturbance_channels is None:
            raise ValueError("H-infinity requires disturbance_channels")
        validate_controller_solution(problem, solution)
        if solution.controller != "h_infinity":
            raise ValueError("solution.controller must be 'h_infinity'")
        if (solution.feasible and solution.gamma_star is None
                and solution.diagnostics.get("synthesis") != "fixed_gamma"):
            raise ValueError("feasible H-infinity solution must report gamma_star")
        return cls(
            gains=solution.gains,
            control_channels=problem.control_channels.to(solution.gains.device),
            feasible=solution.feasible,
            gamma_star=solution.gamma_star,
            diagnostics=solution.diagnostics,
        )

    def reset(self) -> None:
        """Reset online state; finite-horizon state feedback is stateless."""

        return None

    def control(
        self,
        layer_index: int,
        feedback_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return the layer control coordinates for a state deviation."""

        if not self.feasible:
            raise RuntimeError("cannot use an infeasible H-infinity controller")
        gain = self.layer_tensor("gains", self.gains, layer_index, feedback_input)
        return feedback_input @ gain.T

    def solution(self) -> ControllerSolution:
        """Return a serializable copy of the synthesized H-infinity result."""

        return ControllerSolution(
            controller="h_infinity",
            gains=self.gains.detach().cpu(),
            feasible=self.feasible,
            gamma_star=self.gamma_star,
            diagnostics=dict(self.diagnostics),
        )
