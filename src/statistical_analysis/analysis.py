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


# ============================================================================
# ESSENTIAL METRICS TO ANALYZE
# ============================================================================
FACIAL_EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
FACIAL_CORE = ['arousal', 'valence', 'intensity', 'positivity', 'negativity']
VOICE_EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral']
VOICE_CORE = ['arousal', 'valence', 'intensity']
FUSION_CORE = ['fused_arousal', 'fused_valence', 'fused_intensity', 
               'combined_arousal', 'combined_valence', 'combined_intensity']
FUSION_AGREEMENT = ['arousal_modality_agreement', 'valence_modality_agreement', 
                    'intensity_modality_agreement']
SUMMARY_METRICS = ['Overall_Percentage', 'Semantic_Similarity', 
                   'Topic_Coverage', 'Factual_Accuracy']
VOLATILITY_METRICS = ['transition_rate', 'quadrant_transitions']


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


# ============================================================================
# ESSENTIAL GROUP COMPARISONS
# ============================================================================

def load_group_data(data_path: str) -> Dict[str, pd.DataFrame]:
    """Load facial, voice, fusion, and summary data"""
    # Check if we're in the '0 original' folder structure
    if os.path.exists(os.path.join(data_path, '0 original')):
        base_path = os.path.join(data_path, '0 original')
    else:
        base_path = data_path
    
    # CHECK THIS AGAIN
    facial_df = pd.read_csv(os.path.join(base_path, 'correlation', 'facial_summary_merged.csv'))
    voice_df = pd.read_csv(os.path.join(base_path, 'correlation', 'voice_summary_merged.csv'))
    fusion_df = pd.read_csv(os.path.join(base_path, 'correlation', 'fusion_summary_merged.csv'))
    summary_df = pd.read_csv(os.path.join('data\grading_results.csv'))
    
    return {
        'facial': facial_df,
        'voice': voice_df,
        'fusion': fusion_df,
        'summary': summary_df
    }


def compare_groups_on_feature(df: pd.DataFrame, feature: str, group_col: str = 'Group') -> Tuple[float, float, Dict[str, float]]:
    """
    Compare groups on a single feature using Kruskal-Wallis test
    Returns: (test_statistic, p_value, group_means)
    """
    groups = df[group_col].unique()
    samples = []
    group_means = {}
    
    for group in groups:
        group_data = df[df[group_col] == group][feature].dropna()
        if len(group_data) > 0:
            samples.append(group_data.values)
            group_means[group] = float(group_data.mean())
    
    if len(samples) < 2:
        return float('nan'), 1.0, group_means
    
    try:
        stat, p = stats.kruskal(*samples)
    except Exception:
        stat, p = float('nan'), 1.0
    
    return stat, p, group_means


def analyze_facial_emotions(facial_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze facial emotion differences across groups"""
    results = []
    
    # Core emotions
    for emotion in FACIAL_EMOTIONS:
        if f'{emotion}_mean' in facial_df.columns:
            stat, p, means = compare_groups_on_feature(facial_df, f'{emotion}_mean')
            results.append({
                'Category': 'Facial_Emotion',
                'Feature': emotion,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    # Core affective dimensions
    for metric in FACIAL_CORE:
        if f'{metric}_mean' in facial_df.columns:
            stat, p, means = compare_groups_on_feature(facial_df, f'{metric}_mean')
            results.append({
                'Category': 'Facial_Affective',
                'Feature': metric,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    # Volatility metrics
    for metric in VOLATILITY_METRICS:
        if metric in facial_df.columns:
            stat, p, means = compare_groups_on_feature(facial_df, metric)
            results.append({
                'Category': 'Facial_Volatility',
                'Feature': metric,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    return pd.DataFrame(results)


def analyze_voice_emotions(voice_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze voice emotion differences across groups"""
    results = []
    
    # Core emotions
    for emotion in VOICE_EMOTIONS:
        if f'{emotion}_mean' in voice_df.columns:
            stat, p, means = compare_groups_on_feature(voice_df, f'{emotion}_mean')
            results.append({
                'Category': 'Voice_Emotion',
                'Feature': emotion,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    # Core affective dimensions
    for metric in VOICE_CORE:
        if f'{metric}_mean' in voice_df.columns:
            stat, p, means = compare_groups_on_feature(voice_df, f'{metric}_mean')
            results.append({
                'Category': 'Voice_Affective',
                'Feature': metric,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    return pd.DataFrame(results)


def analyze_fusion_emotions(fusion_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze fusion emotion differences across groups"""
    results = []
    
    # Fusion core metrics
    for metric in FUSION_CORE:
        if f'{metric}_mean' in fusion_df.columns:
            stat, p, means = compare_groups_on_feature(fusion_df, f'{metric}_mean')
            results.append({
                'Category': 'Fusion_Affective',
                'Feature': metric,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
        elif metric in fusion_df.columns:  # Try without _mean suffix
            stat, p, means = compare_groups_on_feature(fusion_df, metric)
            results.append({
                'Category': 'Fusion_Affective',
                'Feature': metric,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    # Modality agreement
    for metric in FUSION_AGREEMENT:
        if metric in fusion_df.columns:
            stat, p, means = compare_groups_on_feature(fusion_df, metric)
            results.append({
                'Category': 'Modality_Agreement',
                'Feature': metric,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    # Fusion volatility
    for metric in ['fused_quadrant_transitions', 'combined_quadrant_transitions']:
        if metric in fusion_df.columns:
            stat, p, means = compare_groups_on_feature(fusion_df, metric)
            results.append({
                'Category': 'Fusion_Volatility',
                'Feature': metric,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    return pd.DataFrame(results)


def analyze_summary_quality(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze summary quality differences across groups"""
    results = []
    
    for metric in SUMMARY_METRICS:
        if metric in summary_df.columns:
            stat, p, means = compare_groups_on_feature(summary_df, metric)
            results.append({
                'Category': 'Summary_Quality',
                'Feature': metric,
                'Statistic': stat,
                'P_Value': p,
                'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns',
                **{f'{k}_mean': v for k, v in means.items()}
            })
    
    return pd.DataFrame(results)


def run_essential_analysis(data_path: str = '.') -> pd.DataFrame:
    """
    Run essential group comparison analysis
    Returns comprehensive DataFrame with all results
    """
    print("\n" + "="*80)
    print("ESSENTIAL GROUP COMPARISON ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    data = load_group_data(data_path)
    
    # Run analyses
    print("📊 Analyzing facial emotions...")
    facial_results = analyze_facial_emotions(data['facial'])
    
    print("🎤 Analyzing voice emotions...")
    voice_results = analyze_voice_emotions(data['voice'])
    
    print("🔗 Analyzing fusion emotions...")
    fusion_results = analyze_fusion_emotions(data['fusion'])
    
    print("📝 Analyzing summary quality...")
    summary_results = analyze_summary_quality(data['summary'])
    
    # Combine all results
    all_results = pd.concat([
        facial_results,
        voice_results,
        fusion_results,
        summary_results
    ], ignore_index=True)
    
    # Save results
    output_file = os.path.join(data_path, 'analysis', 'group_comparison_results.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_results.to_csv(output_file, index=False)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_file}")
    print(f"   Total features analyzed: {len(all_results)}")
    print(f"   Significant findings (p<0.05): {len(all_results[all_results['P_Value'] < 0.05])}")
    
    return all_results


def print_summary_report(results: pd.DataFrame):
    """Print a formatted summary of significant findings"""
    print("\n" + "="*80)
    print("SIGNIFICANT GROUP DIFFERENCES (p < 0.05)")
    print("="*80)
    
    sig_results = results[results['P_Value'] < 0.05].sort_values('P_Value')
    
    if len(sig_results) == 0:
        print("\nNo significant differences found.")
        return
    
    for category in sig_results['Category'].unique():
        cat_results = sig_results[sig_results['Category'] == category]
        print(f"\n📊 {category.replace('_', ' ').upper()}:")
        print(f"{'Feature':<30} {'Sig':<5} {'P-Value':<12} {'Group Means'}")
        print("-"*80)
        
        for _, row in cat_results.iterrows():
            means_str = " | ".join([f"{k}: {v:.2f}" for k, v in row.items() 
                                   if k.endswith('_mean') and pd.notna(v)])
            print(f"{row['Feature']:<30} {row['Significant']:<5} {row['P_Value']:<12.6f} {means_str}")
    
    print("\n" + "="*80)
