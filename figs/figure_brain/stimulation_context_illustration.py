#!/usr/bin/env python3
"""Illustrative P10 open-loop / P17 closed-loop residual and accuracy panel."""

from __future__ import annotations

import argparse
import importlib.util
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "brain"
CACHE = HERE / "cache" / "stimulation_context"
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
HORIZONS_MS = (20, 40, 100, 200, 300, 400, 500, 700)
ID_COLOR = "#c9c9c9"
OOD_COLOR = "#8B1E1E"
OPEN_COLOR = "#C47A5A"
CLOSED_COLOR = "#007C7C"
ANALYSIS_VERSION = "artifact-free-stimulation-context-v1"


def load_screen_module():
    path = HERE / "all_subject_conflict_screen.py"
    spec = importlib.util.spec_from_file_location("conflict_screen", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


screen = load_screen_module()


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "pdf.fonttype": 42,
})


CASES = (
    dict(subject=10, baseline_session=2, stim_session=1, lead="LAT", policy="Open-loop"),
    dict(subject=17, baseline_session=1, stim_session=1, lead="LPF", policy="Closed-loop"),
)


def selected_baseline(subject: int, session: int) -> dict:
    options = [item for item in screen.discover_sessions()
               if item["subject"] == subject and item["session"] == session]
    if len(options) != 1:
        raise RuntimeError(f"Expected one P{subject} NoStim{session} session, found {len(options)}")
    return options[0]


def stim_tokens(case: dict, pair_names: list[str], force: bool):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / (f"P{case['subject']}_Stim{case['stim_session']}_"
                          f"{case['lead']}_theta.npz")
    if cache_path.exists() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        if np.isfinite(cached["tokens"]).all() and np.isfinite(cached["baseline"]).all():
            return (cached["tokens"], cached["baseline"],
                    pd.read_csv(io.StringIO(str(cached["trials_csv"]))))

    path = DATA / (f"BIPOLFieldTripFormat_AlignedToImagePresent_"
                   f"P{case['subject']}_Stim{case['stim_session']}.mat")
    loader = screen._load_hdf5_selected if screen.h5py.is_hdf5(path) else screen._load_v5_selected
    data, time, fs, trials = loader(path, pair_names)
    # A subset of legacy FieldTrip files stores time in integer milliseconds.
    if np.nanmax(np.abs(time)) > 100:
        time = time / 1000.0
    valid = ~np.isnan(data).any(axis=(1, 2))
    data, trials = data[valid], trials.loc[valid].reset_index(drop=True)
    tokens, baseline = screen.theta_tokens(data, time, fs)
    np.savez_compressed(cache_path, tokens=tokens, baseline=baseline,
                        trials_csv=trials.to_csv(index=False),
                        pair_names=np.asarray(pair_names), source_name=path.name)
    return tokens, baseline, trials


def compute_context_case(case: dict, force: bool) -> list[dict]:
    selection = selected_baseline(case["subject"], case["baseline_session"])
    tokens, baseline, trials, pair_names = screen.cached_tokens(selection, False)
    lead_columns = [index for index, name in enumerate(pair_names)
                    if screen.re.match(r"^([A-Za-z]+)", name)
                    and screen.re.match(r"^([A-Za-z]+)", name).group(1) == case["lead"]]
    baseline_names = [str(pair_names[index]) for index in lead_columns]
    stim_info = screen.metadata(DATA / (f"BIPOLFieldTripFormat_AlignedToImagePresent_"
                                        f"P{case['subject']}_Stim{case['stim_session']}.mat"))
    stim_clean = set(stim_info["names"]) - stim_info["ictal"]
    common_names = [name for name in baseline_names if name in stim_clean]
    if len(common_names) < 5:
        raise RuntimeError(f"P{case['subject']} {case['lead']}: only {len(common_names)} matched channels")
    base_columns = np.asarray([np.flatnonzero(pair_names == name)[0] for name in common_names])
    base_tokens = tokens[:, :, base_columns]
    base_baseline = baseline[:, base_columns]
    stim_tok, stim_base, stim_trials = stim_tokens(case, common_names, force)

    train, validation, test = screen.split_indices(len(base_tokens))
    mean = base_baseline[train].mean(axis=0)
    std = base_baseline[train].std(axis=0) + 1e-12
    z_base = (base_tokens - mean[None, None, :]) / std[None, None, :]
    z_stim = (stim_tok - mean[None, None, :]) / std[None, None, :]

    # Original artifact-free stimulation-context comparison: train and test ID
    # in the pure no-stimulation session, then test OOD on non-stimulated trials
    # recorded inside the stimulation session.
    stimulation = pd.to_numeric(stim_trials["Stim on/off NEV"], errors="coerce").to_numpy()
    context_indices = np.flatnonzero(stimulation == 0)
    rows = []
    for horizon_ms in HORIZONS_MS:
        steps = horizon_ms // screen.TOKEN_STRIDE_MS
        x_train, y_train = screen.pairs(z_base[train], steps)
        x_validation, y_validation = screen.pairs(z_base[validation], steps)
        a, b, penalty = screen.fit_ridge(x_train, y_train, x_validation, y_validation)
        for split, states in (("ID", z_base[test]), ("OOD", z_stim[context_indices])):
            x, y = screen.pairs(states, steps)
            prediction = x @ a.T + b
            residual = float(np.median(np.sqrt(((y - prediction) ** 2).mean(axis=1))))
            rows.append(dict(subject=f"P{case['subject']}", subject_number=case["subject"],
                             policy=case["policy"], baseline_session=f"NoStim{case['baseline_session']}",
                             context_session=f"Stim{case['stim_session']}", lead=case["lead"],
                             n_channels=len(common_names), horizon_ms=horizon_ms, split=split,
                             linear_residual_rms=residual, ridge_lambda=penalty,
                             n_trials=len(states), analysis_version=ANALYSIS_VERSION))
    return rows


def accuracy_row(subject: int, condition: str, files: list[Path]) -> dict:
    values = []
    for path in files:
        table = screen.metadata(path)["trials"]
        values.extend(pd.to_numeric(table["ResponseAccuracy"], errors="coerce").dropna().tolist())
    values = np.asarray(values)
    correct = int((values == 1).sum())
    n = int(len(values))
    proportion = correct / n
    # Wilson 95% interval for a binomial proportion.
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return dict(subject=f"P{subject}", condition=condition, correct=correct, n=n,
                accuracy_pct=100 * proportion, ci_low_pct=100 * (center - half),
                ci_high_pct=100 * (center + half))


def compute_accuracy() -> pd.DataFrame:
    rows = []
    for case in CASES:
        subject = case["subject"]
        baseline = sorted(DATA.glob(f"BIPOLFieldTripFormat_AlignedToImagePresent_P{subject}_NoStim*.mat"))
        stim = sorted(DATA.glob(f"BIPOLFieldTripFormat_AlignedToImagePresent_P{subject}_Stim*.mat"))
        rows.append(accuracy_row(subject, "No stimulation", baseline))
        rows.append(accuracy_row(subject, case["policy"], stim))
    return pd.DataFrame(rows)


def grouped_bars(axis, frame: pd.DataFrame, title: str, only_ood: bool = False):
    pivot = frame.pivot(index="horizon_ms", columns="split", values="linear_residual_rms").sort_index()
    x = np.arange(len(pivot)); width = 0.36
    if only_ood:
        axis.bar(x, pivot["OOD"], width, color=OOD_COLOR)
    else:
        axis.bar(x - width / 2, pivot["ID"], width, color=ID_COLOR, label="ID")
        axis.bar(x + width / 2, pivot["OOD"], width, color=OOD_COLOR, label="OOD")
    axis.set_xticks(x, [str(value) for value in pivot.index], fontsize=9)
    axis.set_ylim(0, frame["linear_residual_rms"].max() * 1.14)
    axis.set_xlabel("Horizon (ms)", fontsize=11)
    axis.set_ylabel("Linear residual RMS (z)", fontsize=11)
    axis.set_title(title, fontsize=13, weight="bold")
    axis.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def plot(context: pd.DataFrame, accuracy: pd.DataFrame, simplified: bool = False,
         figsize: tuple[float, float] = (4.5, 2.0), output_stem: str | None = None):
    domain = pd.read_csv(RESULTS / "all_subject_conflict_linear_residual.csv")
    domain = domain[(domain["subject_number"] == 10) & (domain["horizon_ms"] <= 200)]
    context = context[context["horizon_ms"] <= 200]
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.34, 0.63])
    axes = np.asarray([
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 4]),
    ])
    grouped_bars(axes[0], domain, "Domain shift")
    grouped_bars(axes[1], context[context["subject_number"] == 10],
                 "Open loop", only_ood=simplified)
    grouped_bars(axes[2], context[context["subject_number"] == 17],
                 "Closed loop", only_ood=simplified)
    residual_max = max(
        domain["linear_residual_rms"].max(),
        context["linear_residual_rms"].max(),
    ) * 1.14
    for axis in axes[:3]:
        axis.set_ylim(0, residual_max)
        axis.title.set_fontsize(9)
        axis.xaxis.label.set_fontsize(8)
        axis.yaxis.label.set_fontsize(8)
        axis.tick_params(axis="both", labelsize=7)
    axes[2].title.set_color(CLOSED_COLOR)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)

    labels = ["Open\nloop", "Closed\nloop"]
    ordered = pd.concat([
        accuracy[(accuracy.subject == "P10") & (accuracy.condition == "Open-loop")],
        accuracy[(accuracy.subject == "P17") & (accuracy.condition == "Closed-loop")],
    ], ignore_index=True)
    colors = [ID_COLOR, CLOSED_COLOR]
    x = np.arange(len(ordered))
    y = ordered["accuracy_pct"].to_numpy()
    errors = np.vstack([y - ordered["ci_low_pct"].to_numpy(),
                        ordered["ci_high_pct"].to_numpy() - y])
    # Keep the established accuracy-panel layout, but make the two bars 70%
    # of their previous width so they no longer dominate the compact axis.
    axes[3].bar(x, y, color=colors, width=0.68 * 0.70)
    if simplified:
        axes[3].errorbar(x[0], y[0], yerr=errors[:, [0]], fmt="none",
                         ecolor="black", capsize=3, lw=1.2)
        axes[3].errorbar(x[1], y[1], yerr=errors[:, [1]], fmt="none",
                         ecolor="black", capsize=3, lw=1.2)
    else:
        axes[3].errorbar(x, y, yerr=errors, fmt="none", ecolor="black", capsize=3, lw=1.2)
    axes[3].set_xticks(x, labels, fontsize=7)
    axes[3].set_ylim(70, 101)
    axes[3].set_ylabel("Accuracy (%)", fontsize=8)
    axes[3].set_title("")
    axes[3].tick_params(axis="y", labelsize=7)
    axes[3].grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
    axes[3].set_axisbelow(True)
    axes[3].spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ID_COLOR, label="ID"),
               plt.Rectangle((0, 0), 1, 1, color=OOD_COLOR, label="OOD")]
    if simplified:
        axes[0].legend(handles=handles, loc="upper left", ncol=2,
                       frameon=False, fontsize=6, handlelength=1.3,
                       columnspacing=0.8, borderaxespad=0.3)
    else:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
                   ncol=2, frameon=False, fontsize=8)
    height_scale = figsize[1] / 2.0
    bottom = 0.34 / height_scale
    top = 1.0 - ((1.0 - 0.82) / height_scale)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=bottom, top=top, wspace=0.12)
    stim_center = (axes[1].get_position().x0 + axes[2].get_position().x1) / 2
    stim_y = top + ((0.955 - 0.82) / height_scale)
    fig.text(stim_center, stim_y, "Stim shift", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="black")
    stem = output_stem or ("stimulation_context_illustration_simplified"
                           if simplified else "stimulation_context_illustration")
    output_dir = PLOTS / "Final Figures" if stem == "figure_stim_context" else PLOTS
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(exist_ok=True); PLOTS.mkdir(exist_ok=True)
    residual_path = RESULTS / "stimulation_context_residual_rms.csv"
    if residual_path.exists() and not args.force:
        context = pd.read_csv(residual_path)
        complete = (set(context["subject_number"]) == {10, 17}
                    and set(context["horizon_ms"]) == set(HORIZONS_MS)
                    and "analysis_version" in context
                    and set(context["analysis_version"]) == {ANALYSIS_VERSION})
    else:
        complete = False
    if not complete:
        rows = []
        for case in CASES:
            print(f"compute P{case['subject']} {case['policy']} context", flush=True)
            rows.extend(compute_context_case(case, args.force))
        context = pd.DataFrame(rows)
        context.to_csv(residual_path, index=False)
    else:
        print(f"reuse fitted residual cache: {residual_path}", flush=True)

    accuracy = compute_accuracy()
    accuracy.to_csv(RESULTS / "stimulation_context_accuracy.csv", index=False)
    plot(context, accuracy)
    plot(context, accuracy, simplified=True)
    plot(context, accuracy, figsize=(5.0, 4.0), output_stem="figure_stim_context")
    print(context.pivot_table(index=["subject", "policy", "horizon_ms"],
                              columns="split", values="linear_residual_rms").to_string())
    print("\n", accuracy.to_string(index=False))
    print(PLOTS / "stimulation_context_illustration.png")


if __name__ == "__main__":
    main()
