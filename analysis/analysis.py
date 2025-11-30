import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None


def participant_mean_metrics(resampled: pd.DataFrame) -> Dict[str, float]:
    metrics = {}
    for col in ("fused_arousal", "fused_valence", "fused_intensity", "combined_arousal", "combined_valence", "combined_intensity"):
        if col in resampled.columns:
            metrics[col] = float(np.nanmean(resampled[col].values.astype(float)))
    return metrics


def group_summary_resampled(group_resampled: List[pd.DataFrame]) -> pd.DataFrame:
    if len(group_resampled) == 0:
        return pd.DataFrame()
    # align columns by intersecting numeric columns
    cols = set.intersection(*(set(df.columns) for df in group_resampled))
    # pick numeric cols
    numeric_cols = [c for c in cols if np.issubdtype(group_resampled[0][c].dtype, np.number)]
    arr = np.stack([df[numeric_cols].to_numpy() for df in group_resampled], axis=0)  # participants x segments x features
    mean = np.nanmean(arr, axis=0)
    sem = stats.sem(arr, axis=0, nan_policy='omit')
    seg = np.arange(mean.shape[0])
    rows = []
    for i in range(mean.shape[0]):
        row = {f"{col}_mean": mean[i, j] for j, col in enumerate(numeric_cols)}
        row.update({f"{col}_sem": sem[i, j] for j, col in enumerate(numeric_cols)})
        row["segment"] = i
        rows.append(row)
    return pd.DataFrame(rows).set_index("segment")


def timepoint_kruskal(groups_resampled: Dict[str, List[pd.DataFrame]], feature: str) -> Tuple[np.ndarray, np.ndarray]:
    # determine number of segments from first available participant
    n = None
    for g in groups_resampled:
        if groups_resampled[g]:
            n = groups_resampled[g][0].shape[0]
            break
    if n is None:
        return np.array([]), np.array([])
    pvals = np.ones(n)
    for i in range(n):
        samples = []
        for g, lst in groups_resampled.items():
            # collect value at segment i for each participant
            vals = [df[feature].iloc[i] for df in lst if feature in df.columns]
            if len(vals) > 0:
                samples.append(vals)
        if len(samples) >= 2:
            try:
                stat, p = stats.kruskal(*samples)
            except Exception:
                p = 1.0
            pvals[i] = p
        else:
            pvals[i] = 1.0

    if multipletests is not None:
        reject, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    else:
        # fallback: simple Bonferroni
        pvals_corr = pvals * len(pvals)
        reject = pvals_corr < 0.05
    return pvals_corr, reject


def overall_group_tests(groups_resampled: Dict[str, List[pd.DataFrame]], feature: str) -> Tuple[float, float]:
    group_values = []
    for g, lst in groups_resampled.items():
        vals = []
        for df in lst:
            if feature in df.columns:
                vals.append(float(np.nanmean(df[feature].values.astype(float))))
        if vals:
            group_values.append(vals)
    if len(group_values) < 2:
        return float('nan'), 1.0
    try:
        stat, p = stats.kruskal(*group_values)
    except Exception:
        stat, p = float('nan'), 1.0
    return stat, p


def dominant_emotion_distribution(groups_resampled: Dict[str, List[pd.DataFrame]]) -> Dict[str, Counter]:
    out = {}
    for g, lst in groups_resampled.items():
        cnt = Counter()
        total = 0
        for df in lst:
            if 'dominant_emotion' in df.columns:
                vals = df['dominant_emotion'].astype(str).tolist()
                cnt.update(vals)
                total += len(vals)
        out[g] = cnt
    return out


def emotion_change_rate(df: pd.DataFrame) -> float:
    if 'dominant_emotion' not in df.columns:
        return np.nan
    arr = df['dominant_emotion'].astype(str).to_numpy()
    if len(arr) < 2:
        return 0.0
    changes = np.sum(arr[1:] != arr[:-1])
    return float(changes) / float(len(arr))


def group_change_rates(groups_resampled: Dict[str, List[pd.DataFrame]]) -> Dict[str, List[float]]:
    out = {}
    for g, lst in groups_resampled.items():
        rates = [emotion_change_rate(df) for df in lst]
        out[g] = rates
    return out


def group_emotion_journey(groups_resampled: Dict[str, List[pd.DataFrame]]) -> Dict[str, List[str]]:
    journeys = {}
    for g, lst in groups_resampled.items():
        if not lst:
            journeys[g] = []
            continue
        n = lst[0].shape[0]
        seq = []
        for i in range(n):
            labels = [df['dominant_emotion'].iloc[i] for df in lst if 'dominant_emotion' in df.columns]
            if labels:
                vals, counts = np.unique(labels, return_counts=True)
                seq.append(vals[np.argmax(counts)])
            else:
                seq.append("")
        journeys[g] = seq
    return journeys
