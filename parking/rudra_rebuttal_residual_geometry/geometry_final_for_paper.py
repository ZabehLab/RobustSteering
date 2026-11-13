"""Compact paper variant of the final geometry figure; uses the same cached data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from .performance_grid import TEAL, GRAY
from .pipeline import ROOT, UNIT, SEED, read, save, sha

METHODS = ('original', 'lqr_8d', 'empirical', 'rotated', 'isotropic')
LABELS = ('Original', 'LQR (8D)', 'H∞ (Empirical)', 'H∞ (Rand. rot.)', 'H∞ (Isotropic)')


def draw_alignment(fig, ax, colorbar_axis):
    alignment = read(ROOT / 'alignment_n050.json')
    comparisons = {r['alternative']: r for r in
                   read(ROOT / 'alignment_comparisons_n050.json')['comparisons']}
    variants = METHODS[2:]
    values = [alignment['alignment'][v] for v in variants]
    layers = np.asarray(alignment['layers'])
    jitter = np.random.default_rng(SEED + 3).uniform(-.075, .075, size=len(layers))
    boundaries = np.arange(layers.min() - .5, layers.max() + 1.5)
    cmap = matplotlib.colormaps['viridis'].resampled(len(layers))
    norm = BoundaryNorm(boundaries, cmap.N)
    ax.add_patch(Rectangle((-.35, 0), .7, 1, facecolor=to_rgba('#008080', .14),
                          edgecolor='none', zorder=.1))
    ax.boxplot(values, positions=np.arange(3), widths=.28, patch_artist=True,
               whis=(0, 100), showfliers=False, manage_ticks=False,
               boxprops={'facecolor': to_rgba('#777777', .7), 'edgecolor': 'none', 'linewidth': 0},
               medianprops={'color': '#202427', 'linewidth': 1.4},
               whiskerprops={'color': '#60666b'}, capprops={'color': '#60666b'})
    for position, points in enumerate(values):
        dots = ax.scatter(position + jitter, points, c=layers, cmap=cmap, norm=norm,
                          s=24, edgecolors='none', linewidths=0, zorder=3)
    ax.set(xticks=np.arange(3), xticklabels=LABELS[2:],
           ylabel='OOD cov. alignment\nw/ calib.', ylim=(0, 1.13))
    ax.set_yticks(np.linspace(0, 1, 6))
    for endpoint, y, alternative in zip((1, 2), (1.03, 1.08), variants[1:], strict=True):
        ax.plot([0, 0, endpoint, endpoint], [y - .012, y, y, y - .012],
                color='#45494d', linewidth=1.4)
        ax.text(endpoint / 2, y, comparisons[alternative]['stars'], ha='center', va='center',
                fontsize=9, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .2})
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor')
    ax.get_xticklabels()[0].set_color(TEAL)
    ax.get_xticklabels()[0].set_fontweight('bold')
    ax.tick_params(axis='both', direction='out', length=3.5, width=.8, pad=5)
    ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
    ax.set_axisbelow(True)
    colorbar = fig.colorbar(dots, cax=colorbar_axis, boundaries=boundaries, drawedges=False)
    colorbar.set_label('Qwen2.5-0.5B-\nInstruct layers')
    colorbar.set_ticks([1, 6, 12, 18, 24])
    colorbar.minorticks_off()
    colorbar.outline.set_visible(False)
    colorbar.solids.set_edgecolor('face')


def report():
    performance = ROOT / 'paper_axbench_n200/pooled_results.json'
    payload = read(performance)
    assert payload['judge'] == 'gpt-4o-mini-2024-07-18'
    lookup = {r['method']: r for r in payload['results']}
    rows = [lookup[m] for m in METHODS]
    assert all(r['count_id'] == r['count_ood'] == 200 and r['count_total'] == 400 for r in rows)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 12, 'axes.labelsize': 12,
                         'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8, 'axes.edgecolor': '#454545', 'pdf.fonttype': 42}):
        # Preserve horizontal placement and room for the rotated category labels.
        figure_height = 3.8 * .70
        bottom = (3.8 * .33) / figure_height
        height = 1 - bottom - (3.8 * .06) / figure_height
        width = 4.8 * (.98 - .17) / (2 + .55)
        alignment_width = width * .6
        saved_width = width - alignment_width
        figure_width = 8.0 - saved_width
        fig = plt.figure(figsize=(figure_width, figure_height))
        axes = [fig.add_axes([x / figure_width, bottom, width / figure_width, height])
                for x in (.816, .816 + width * 1.55)]
        alignment_x = .816 + 2 * width * 1.55
        alignment_axis = fig.add_axes([alignment_x / figure_width, bottom,
                                      alignment_width / figure_width, height])
        colorbar_axis = fig.add_axes([(alignment_x + alignment_width + .144) / figure_width,
                                     bottom, .08 / figure_width, height])
        draw_alignment(fig, alignment_axis, colorbar_axis)
        for ax, (metric, label) in zip(axes, [('asr_pct', 'Attack\nsuccess rate (%) ↓'),
                                                ('overall', 'AXBench overall ↑')], strict=True):
            ax.bar(np.arange(len(METHODS)), [r[metric] for r in rows], width=.40,
                   color=[TEAL if m == 'empirical' else GRAY for m in METHODS],
                   edgecolor='none', zorder=3)
            ax.set_xticks(np.arange(len(METHODS)), LABELS)
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right', rotation_mode='anchor')
            ax.set_xlim(-.6, len(METHODS) - .4)
            ax.tick_params(axis='both', direction='out', length=3.5, width=.8, pad=5)
            ax.grid(axis='y', color='#e5e7eb', linewidth=.7)
            ax.set_axisbelow(True)
            ax.set_ylim(0, max(r[metric] for r in rows) * 1.10)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, integer=metric == 'asr_pct'))
            ax.set_ylabel(label)
            tick = ax.get_xticklabels()[METHODS.index('empirical')]
            tick.set_color(TEAL)
            tick.set_fontweight('bold')
        for extension in ('png', 'pdf'):
            fig.savefig(UNIT / f'geometry_final_for_paper.{extension}', dpi=600)
        plt.close(fig)
    inputs = [ROOT / 'alignment_n050.json', ROOT / 'alignment_comparisons_n050.json', performance]
    save(ROOT / 'geometry_final_for_paper_inputs.json', {
        'input_sha256': {str(p): sha(p) for p in inputs},
        'alignment_behaviors': 50, 'performance_id_responses': 200, 'performance_ood_responses': 200,
        'performance_judge': payload['judge'], 'method_order_performance': list(METHODS),
    })
    print('Saved compact paper geometry 1×3 figure using cached alignment and pooled paper-judge performance.')


if __name__ == '__main__':
    report()
