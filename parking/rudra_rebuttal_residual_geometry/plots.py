"""Two requested figures and a compact numeric summary for this pilot."""

import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, to_rgba
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from .geometry import VARIANTS, ACTIVE_VARIANTS
from .pipeline import ROOT, UNIT, SCORER, SEED, read, save, sample, blocks, generation_path, tensor_read
from robust_steerability.judges.specs import scorer_cache_path

LABELS = ['Empirical', 'Isotropic', 'Diagonal', 'Randomly\nrotated']
TEAL, RED = '#008080', '#921f25'


def report(count):
    rng = np.random.default_rng(SEED + 2)
    bootstrap = rng.integers(0, count, size=(2000, count))
    summary, estimates = [], {}
    for variant in VARIANTS:
        controller = tensor_read(ROOT / 'controllers' / f'{variant}.pt')
        row = {'geometry': variant, 'gamma_star': controller['gamma_star'],
               'gamma_used': controller['diagnostics']['gamma_used'], 'behaviors_per_condition': count}
        for condition in ('id', 'ood'):
            scores = {}
            for index, _ in blocks(count, condition):
                path = scorer_cache_path(ROOT, generation_path(variant, condition, index), SCORER)
                payload = read(path)
                assert payload['status'] == 'complete' and all(r['valid'] for r in payload['rows'])
                scores.update({r['prompt_id']: r['score'] for r in payload['rows']})
            ids = [pair[condition]['prompt_id'] for pair in sample(count)]
            assert set(scores) == set(ids)
            values = np.array([scores[i] for i in ids], dtype=float)
            estimates[variant, condition] = values[bootstrap].mean(1) * 100
            row[condition + '_asr_pct'] = values.mean() * 100
            row[condition + '_jackknife_se_pct'] = np.sqrt(values.mean() * (1-values.mean()) / (count-1)) * 100
        summary.append(row)
    save(ROOT / f'results_n{count:03d}.json', {'results': summary,
         'boxplot': '2000 paired behavior-bootstrap ASR estimates; not independent model runs',
         'rotation': 'one predefined random rotation; exploratory pilot',
         'calibration': 'reused frozen Mega fit; no retuning', 'sampling': 'one jailbreak per behavior; 10 per template at n=50'})
    with (UNIT / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 13,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for condition, offset, color in [('id', -.18, '#595f65'), ('ood', .18, RED)]:
        distributions = [estimates[v, condition] for v in VARIANTS]
        positions = np.arange(4) + offset
        boxes = ax.boxplot(distributions, positions=positions, widths=.28, patch_artist=True,
                           showfliers=False, manage_ticks=False,
                           medianprops={'color': color, 'linewidth': 1.5},
                           whiskerprops={'color': color}, capprops={'color': color})
        for patch in boxes['boxes']:
            patch.set(facecolor=color, edgecolor=color, alpha=.25)
        ax.scatter(positions, [r[condition + '_asr_pct'] for r in summary], color=color, s=25, zorder=5)
    ax.set(xticks=np.arange(4), xticklabels=LABELS, ylabel='Attack success rate (%)',
           title='H∞ steering under matched-scale disturbance geometries')
    ax.set_ylim(bottom=min(-1, ax.get_ylim()[0]))
    ax.legend(handles=[Patch(facecolor='#595f65', alpha=.5, label='ID: direct'),
                       Patch(facecolor=RED, alpha=.5, label='OOD: pooled jailbreaks')], frameon=False, loc='upper right')
    ax.grid(axis='y', alpha=.12)
    fig.text(.5, .015, f'{count} paired behaviors · boxes: paired bootstrap ASR estimates', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(UNIT / f'geometry_behavior.{ext}', dpi=220)
    plt.close(fig)
    plot_alignment(count)
    print('Saved geometry_behavior and covariance_alignment (PNG/PDF), plus results.csv.', flush=True)


def plot_alignment(count):
    alignment = read(ROOT / f'alignment_n{count:03d}.json')
    comparison_path = ROOT / f'alignment_comparisons_n{count:03d}.json'
    if not comparison_path.exists():
        from .statistics import alignment_comparisons
        from .pipeline import tensor_save
        residuals = tensor_read(ROOT / 'residuals' / f'ood_n{count:03d}.pt')['residuals'].numpy()
        fit = tensor_read(ROOT / 'frozen_fit.pt')
        comparisons, bootstrap = alignment_comparisons(
            residuals, np.stack([fit['covariances'][v].numpy() for v in VARIANTS]), SEED + 4)
        save(comparison_path, comparisons)
        import torch
        tensor_save(ROOT / f'alignment_bootstrap_n{count:03d}.pt',
                    {'mean_alignment_differences': torch.from_numpy(bootstrap)})
    comparison_lookup = {r['alternative']: r for r in read(comparison_path)['comparisons']}
    comparisons = [comparison_lookup[v] for v in ACTIVE_VARIANTS if v != 'empirical']
    layers = np.asarray(alignment['layers'])
    values = [alignment['alignment'][variant] for variant in ACTIVE_VARIANTS]
    jitter = np.random.default_rng(SEED + 3).uniform(-.075, .075, size=len(layers))
    boundaries = np.arange(layers.min() - .5, layers.max() + 1.5)
    cmap = matplotlib.colormaps['viridis'].resampled(len(layers))
    norm = BoundaryNorm(boundaries, cmap.N)
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 16,
                         'axes.labelsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.add_patch(Rectangle((-.35, 0), .7, 1, facecolor=to_rgba(TEAL, .14),
                               edgecolor='none', zorder=.1))
        boxes = ax.boxplot(values, positions=np.arange(len(ACTIVE_VARIANTS)), widths=.28, patch_artist=True,
                   whis=(0, 100), showfliers=False, manage_ticks=False,
                   boxprops={'facecolor': 'white', 'edgecolor': 'none', 'linewidth': 0},
                   medianprops={'color': '#202427', 'linewidth': 1.4},
                   whiskerprops={'color': '#60666b'}, capprops={'color': '#60666b'})
        for box in boxes['boxes']:
            box.set_facecolor(to_rgba('#777777', .7))
        for position, points in enumerate(values):
            dots = ax.scatter(position + jitter, points, c=layers, cmap=cmap,
                              norm=norm, s=24,
                              edgecolors='none', linewidths=0, zorder=3)
        ax.set(xticks=np.arange(len(ACTIVE_VARIANTS)), xticklabels=['Empirical', 'Isotropic', 'Rotated'],
               ylabel='OOD alignment w/ calibration', ylim=(0, 1.13))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.spines['left'].set_color('#777777')
        for endpoint, y, comparison in zip((1, 2), (1.03, 1.08), comparisons):
            ax.plot([0, 0, endpoint, endpoint], [y - .012, y, y, y - .012],
                    color='#45494d', linewidth=1.4)
            ax.text(endpoint / 2, y, comparison['stars'],
                    ha='center', va='center', fontsize=9,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .2})
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', pad=1.5)
        colorbar_axis = make_axes_locatable(ax).append_axes('right', size='5%', pad=.144)
        colorbar = fig.colorbar(dots, cax=colorbar_axis,
                               boundaries=boundaries, drawedges=False)
        colorbar.set_label('Qwen2.5-0.5B-Instruct layers')
        colorbar.set_ticks([1, 6, 12, 18, 24])
        colorbar.minorticks_off()
        colorbar.outline.set_visible(False)
        colorbar.solids.set_edgecolor('face')
        ax.grid(axis='y', alpha=.12)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.canvas.draw()
        plot_bounds = ax.get_position().frozen()
        bar_bounds = colorbar_axis.get_position().frozen()
        old_width, height = fig.get_size_inches()
        new_width = old_width * (1 - .4 * plot_bounds.width)
        scale = old_width / new_width
        # Keep text margins, height and color-bar width; compress only the data axis.
        ax.set_axes_locator(None)
        colorbar_axis.set_axes_locator(None)
        fig.set_size_inches(new_width, height)
        ax.set_position([plot_bounds.x0 * scale, plot_bounds.y0,
                         plot_bounds.width * .6 * scale, plot_bounds.height])
        colorbar_axis.set_position([(bar_bounds.x0 - .4 * plot_bounds.width) * scale,
                                   bar_bounds.y0, bar_bounds.width * scale, bar_bounds.height])
        for ext in ('png', 'pdf'):
            fig.savefig(UNIT / f'covariance_alignment.{ext}', dpi=220)
        plt.close(fig)

