"""Runner script to perform group-level multimodal emotion analysis.

Usage: python analysis/run_analysis.py --results-dir results --out-dir analysis_results
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from statistical_analysis import loader, normalizer, analysis_module


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def main(results_dir: str, out_dir: str, n_segments: int = 100):
    ensure_dir(out_dir)
    groups = loader.load_all_groups(results_dir)

    # resample each participant to fixed segments
    groups_resampled = {}
    for g, items in groups.items():
        resampled = []
        for pid, df in items:
            rs = normalizer.resample_participant(df, n_points=n_segments)
            rs.index.name = 'segment'
            rs['participant'] = pid
            resampled.append(rs)
        groups_resampled[g] = resampled

    # Save per-group counts
    summary = {g: len(lst) for g, lst in groups_resampled.items()}
    with open(os.path.join(out_dir, 'groups_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Timepoint tests for fused_arousal/fused_valence/fused_intensity if available
    features = ['fused_arousal', 'fused_valence', 'fused_intensity', 'combined_arousal', 'combined_valence', 'combined_intensity']
    results_tests = {}
    for feat in features:
        pvals, reject = analysis_module.timepoint_kruskal(groups_resampled, feat)
        if pvals.size:
            results_tests[feat] = {
                'pvals': pvals.tolist(),
                'significant_segments': [int(i) for i, r in enumerate(reject) if r]
            }
            # also plot group means for visual inspection
            plt.figure(figsize=(8, 4))
            for g, lst in groups_resampled.items():
                if not lst or feat not in lst[0].columns:
                    continue
                arr = pd.DataFrame([df[feat].values for df in lst])
                mean = arr.mean(axis=0)
                plt.plot(mean, label=g)
            plt.title(feat)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{feat}_group_means.png'))
            plt.close()

    with open(os.path.join(out_dir, 'timepoint_tests.json'), 'w', encoding='utf-8') as f:
        json.dump(results_tests, f, indent=2)

    # Overall group tests on participant means
    overall = {}
    for feat in features:
        stat, p = analysis_module.overall_group_tests(groups_resampled, feat)
        overall[feat] = {'stat': float(stat) if not pd.isna(stat) else None, 'pvalue': float(p)}
    with open(os.path.join(out_dir, 'overall_group_tests.json'), 'w', encoding='utf-8') as f:
        json.dump(overall, f, indent=2)

    # Dominant emotion distributions
    dist = analysis_module.dominant_emotion_distribution(groups_resampled)
    # convert Counters to dicts
    dist_out = {g: dict(c) for g, c in dist.items()}
    with open(os.path.join(out_dir, 'dominant_emotion_distribution.json'), 'w', encoding='utf-8') as f:
        json.dump(dist_out, f, indent=2)

    # Change rates
    rates = analysis_module.group_change_rates(groups_resampled)
    with open(os.path.join(out_dir, 'change_rates.json'), 'w', encoding='utf-8') as f:
        json.dump(rates, f, indent=2)

    # Emotion journey
    journeys = analysis_module.group_emotion_journey(groups_resampled)
    with open(os.path.join(out_dir, 'group_emotion_journeys.json'), 'w', encoding='utf-8') as f:
        json.dump(journeys, f, indent=2)

    print('Analysis complete. Outputs written to', out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out-dir', default='analysis_results')
    ap.add_argument('--n-segments', type=int, default=100)
    args = ap.parse_args()
    main(args.results_dir, args.out_dir, n_segments=args.n_segments)
