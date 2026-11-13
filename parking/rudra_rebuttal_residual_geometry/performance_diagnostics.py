"""Separate ID/OOD diagnostics with the explicitly requested scoring sources."""

import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from robust_steerability.judges.exact import harmonic_mean
from .pipeline import ROOT, UNIT, save
from .score_store import METHODS, LABELS, load_cases, prepare_baselines

SOURCES = {method: ('paper_judge' if method in ('original', 'lqr_8d') else 'assistant_audit')
           for method in METHODS}
METRICS = [('asr_pct', 'ASR ↓'), ('concept_relevance', 'Safe-concept relevance ↑'),
           ('instruction_relevance', 'Instruction relevance ↑'),
           ('fluency', 'Fluency ↑'), ('overall', 'AXBench overall ↑')]


def report():
    prepare_baselines(50)
    results = []
    for condition in ('id', 'ood'):
        for method in METHODS:
            rows = load_cases(method, condition, score_source=SOURCES[method])
            assert len(rows) == 50
            raw = {key: np.array([r[key] for r in rows], dtype=float)
                   for key in ('concept_relevance', 'instruction_relevance', 'fluency')}
            raw['asr_pct'] = np.array([100 * r['harmbench_attack_success'] for r in rows])
            raw['overall'] = np.array([harmonic_mean([r['concept_relevance'], r['instruction_relevance'], r['fluency']])
                                       for r in rows])
            item = {'method': method, 'condition': condition, 'count': len(rows),
                    'axbench_score_source': SOURCES[method], 'asr_score_source': 'cached HarmBench test classifier'}
            for metric, values in raw.items():
                item[metric] = float(values.mean())
                item[metric + '_se'] = float(values.std(ddof=1) / np.sqrt(len(rows)))
            results.append(item)
    save(ROOT / 'diagnostics/performance_with_baselines.json', {
        'results': results, 'axbench_sources': SOURCES,
        'scope': 'Diagnostic with mixed AXBench grading sources, explicitly requested by the user; not a uniform-judge comparison',
        'asr': 'Same cached official HarmBench test classifier for all methods',
        'overall': 'Mean of per-response harmonic means; no new joint-success metric',
    })
    with (UNIT / 'geometry_performance.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(results)
    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42}):
        for condition, color, title in [('id', '#62686e', 'ID — direct prompts'),
                                        ('ood', '#921f25', 'OOD — pooled jailbreaks')]:
            selected = [next(r for r in results if r['method'] == method and r['condition'] == condition)
                        for method in METHODS]
            fig, axes = plt.subplots(1, 5, figsize=(19, 4.5))
            for ax, (metric, heading) in zip(axes, METRICS):
                ax.bar(np.arange(len(METHODS)), [r[metric] for r in selected], width=.55,
                       color=color, alpha=.7, edgecolor='none', capsize=2,
                       yerr=[r[metric + '_se'] for r in selected],
                       error_kw={'elinewidth': 1, 'capthick': 1, 'ecolor': color})
                ax.set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS],
                              rotation=45, ha='right', rotation_mode='anchor')
                ax.set_title(heading, fontweight='bold')
                ax.set_ylim(bottom=0)
                if metric != 'asr_pct':
                    ax.set_ylim(0, 2.1)
                    ax.set_yticks([0, 1, 2])
                ax.grid(axis='y', alpha=.12)
                ax.set_axisbelow(True)
            axes[0].set_ylabel('Attack success rate (%)')
            axes[1].set_ylabel('Score (0–2)')
            fig.suptitle(title, fontsize=14, fontweight='bold', y=.98)
            fig.text(.5, .015,
                     'Original & LQR (8D): cached GPT-4o-mini scores.  H∞ arms: assistant-audit scores.  ASR: cached HarmBench for all.',
                     ha='center', fontsize=9, color='#45494d')
            fig.tight_layout(rect=(0, .07, 1, .94), w_pad=2)
            for extension in ('png', 'pdf'):
                fig.savefig(UNIT / f'geometry_performance_{condition}.{extension}', dpi=220)
            plt.close(fig)
    for item in results:
        print(item['method'], item['condition'], item['axbench_score_source'],
              ' | '.join(f'{key}={item[key]:.3f}' for key, _ in METRICS))


if __name__ == '__main__':
    report()
