"""Official HarmBench ASR summaries for the active geometry performance arms."""

import argparse
import json
from math import sqrt

from .pipeline import ROOT, save
from .score_store import ACTIVE_METHODS, load_responses


def asr_results(count):
    results = []
    for condition in ('id', 'ood'):
        for method in ACTIVE_METHODS:
            rows = load_responses(method, condition, count)
            values = [r['harmbench_attack_success'] for r in rows]
            assert len(values) == count and all(v in (0, 1) for v in values)
            rate = sum(values) / count
            results.append({'method': method, 'condition': condition, 'count': count,
                            'successful_attacks': sum(values), 'asr_pct': rate * 100,
                            'asr_pct_se': 100 * sqrt(rate * (1-rate) / (count-1)),
                            'asr_score_source': 'cached HarmBench test classifier'})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=200)
    count = parser.parse_args().count
    payload = {'results': asr_results(count)}
    save(ROOT / f'asr_performance_n{count:03d}.json', payload)
    print(json.dumps(payload, indent=2))
