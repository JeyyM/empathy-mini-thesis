#!/usr/bin/env python3
"""
Correlate fusion (combined facial+voice) emotion data with summary grading results
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_summary_grades():
    """Load the summary grading results"""
    grades_path = "../text summarization 2/grading_results.csv"
    df = pd.read_csv(grades_path)
    return df

def extract_fusion_stats(group, name):
    """Extract comprehensive fusion emotion statistics from the fusion CSV"""
    # Construct the file path
    base_path = f"../results/{group.capitalize()}/{name}"
    csv_file = f"Final{name}_ml_fusion.csv"
    file_path = os.path.join(base_path, csv_file)
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return None
    
    # Load the fusion data
    df = pd.read_csv(file_path)
    
    # Extract fusion columns
    fusion_cols = {
        # Fused emotions
        'angry': 'fused_angry',
        'disgust': 'fused_disgust',
        'fear': 'fused_fear',
        'happy': 'fused_happy',
        'sad': 'fused_sad',
        'surprise': 'fused_surprise',
        'neutral': 'fused_neutral',
        
        # Fused dimensions
        'arousal': 'fused_arousal',
        'valence': 'fused_valence',
        'intensity': 'fused_intensity',
        'stress': 'fused_stress',
        'positivity': 'fused_positivity',
        'negativity': 'fused_negativity',
        'excitement': 'fused_excitement',
        'calmness': 'fused_calmness',
        
        # Combined dimensions (from facial + voice averaging)
        'combined_arousal': 'combined_arousal',
        'combined_valence': 'combined_valence',
        'combined_intensity': 'combined_intensity'
    }
    
    stats_dict = {
        'group': group,
        'name': name
    }
    
    # Basic statistics for each fusion metric
    for metric_name, col_name in fusion_cols.items():
        if col_name in df.columns:
            data = df[col_name].dropna()
            
            if len(data) == 0:
                continue
                
            stats_dict[f'{metric_name}_mean'] = data.mean()
            stats_dict[f'{metric_name}_std'] = data.std()
            stats_dict[f'{metric_name}_min'] = data.min()
            stats_dict[f'{metric_name}_max'] = data.max()
            stats_dict[f'{metric_name}_median'] = data.median()
            
            # VOLATILITY METRICS
            stats_dict[f'{metric_name}_range'] = data.max() - data.min()
            stats_dict[f'{metric_name}_variance'] = data.var()
            stats_dict[f'{metric_name}_cv'] = data.std() / abs(data.mean()) if data.mean() != 0 else 0
            
            # CHANGE METRICS
            if len(data) > 1:
                changes = data.diff().abs()
                stats_dict[f'{metric_name}_mean_change'] = changes.mean()
                stats_dict[f'{metric_name}_max_change'] = changes.max()
                stats_dict[f'{metric_name}_total_change'] = changes.sum()
            
            # TREND METRICS
            if len(data) > 2:
                x = np.arange(len(data))
                try:
                    slope, intercept = np.polyfit(x, data, 1)
                    stats_dict[f'{metric_name}_trend_slope'] = slope
                    stats_dict[f'{metric_name}_trend_direction'] = 1 if slope > 0 else -1 if slope < 0 else 0
                except:
                    pass
    
    # FUSED QUADRANT TIME ANALYSIS
    if 'fused_quadrant' in df.columns:
        quadrant_counts = df['fused_quadrant'].value_counts()
        total_frames = len(df)
        
        for quadrant in ['Happy', 'Calm', 'Stressed', 'Tired']:
            count = quadrant_counts.get(quadrant, 0)
            stats_dict[f'fused_time_in_{quadrant.lower()}'] = count
            stats_dict[f'fused_pct_time_{quadrant.lower()}'] = (count / total_frames) * 100 if total_frames > 0 else 0
        
        # TRANSITION FREQUENCY
        if len(df) > 1:
            transitions = (df['fused_quadrant'] != df['fused_quadrant'].shift()).sum()
            stats_dict['fused_quadrant_transitions'] = transitions
            stats_dict['fused_transition_rate'] = transitions / len(df) if len(df) > 0 else 0
    
    # COMBINED QUADRANT ANALYSIS (from facial+voice averaging)
    if 'combined_quadrant' in df.columns:
        quadrant_counts = df['combined_quadrant'].value_counts()
        total_frames = len(df)
        
        for quadrant in ['Happy', 'Calm', 'Stressed', 'Tired']:
            count = quadrant_counts.get(quadrant, 0)
            stats_dict[f'combined_time_in_{quadrant.lower()}'] = count
            stats_dict[f'combined_pct_time_{quadrant.lower()}'] = (count / total_frames) * 100 if total_frames > 0 else 0
        
        if len(df) > 1:
            transitions = (df['combined_quadrant'] != df['combined_quadrant'].shift()).sum()
            stats_dict['combined_quadrant_transitions'] = transitions
            stats_dict['combined_transition_rate'] = transitions / len(df) if len(df) > 0 else 0
    
    # DOMINANT FUSED EMOTION ANALYSIS
    emotion_cols = ['fused_happy', 'fused_angry', 'fused_sad', 'fused_fear', 
                   'fused_surprise', 'fused_disgust', 'fused_neutral']
    
    available_emotion_cols = [col for col in emotion_cols if col in df.columns]
    
    if len(available_emotion_cols) > 0:
        emotion_means = {}
        for col in available_emotion_cols:
            data = df[col].dropna()
            if len(data) > 0:
                emotion_means[col.replace('fused_', '')] = data.mean()
        
        if emotion_means:
            stats_dict['fused_dominant_emotion'] = max(emotion_means, key=emotion_means.get)
            stats_dict['fused_dominant_emotion_strength'] = max(emotion_means.values())
            
            # Emotional diversity
            emotion_values = list(emotion_means.values())
            total = sum(emotion_values)
            if total > 0:
                proportions = [v/total for v in emotion_values]
                diversity = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
                stats_dict['fused_emotional_diversity'] = diversity
    
    # OVERALL FUSED EMOTIONAL STABILITY
    core_fusion = ['fused_valence', 'fused_arousal', 'fused_intensity']
    available_core = [col for col in core_fusion if col in df.columns]
    
    if len(available_core) > 0:
        stds = []
        for col in available_core:
            data = df[col].dropna()
            if len(data) > 0:
                stds.append(data.std())
        
        if stds:
            combined_std = sum(stds) / len(stds)
            stats_dict['fused_overall_stability'] = 1 / (1 + combined_std)
            stats_dict['fused_overall_volatility'] = combined_std
    
    # PEAK FUSED INTENSITY MOMENTS
    if 'fused_intensity' in df.columns:
        intensity_data = df['fused_intensity'].dropna()
        if len(intensity_data) > 0:
            top_10_pct = max(1, int(len(intensity_data) * 0.1))
            top_intensities = intensity_data.nlargest(top_10_pct)
            stats_dict['fused_peak_intensity_mean'] = top_intensities.mean()
            stats_dict['fused_peak_intensity_max'] = top_intensities.max()
    
    # MODALITY AGREEMENT - How well facial and voice agree
    if all(col in df.columns for col in ['facial_valence', 'voice_valence']):
        facial_val = df['facial_valence'].dropna()
        voice_val = df['voice_valence'].dropna()
        
        if len(facial_val) > 0 and len(voice_val) > 0:
            # Correlation between facial and voice valence
            min_len = min(len(facial_val), len(voice_val))
            if min_len > 2:
                corr, _ = stats.spearmanr(facial_val[:min_len], voice_val[:min_len])
                stats_dict['valence_modality_agreement'] = corr
    
    if all(col in df.columns for col in ['facial_arousal', 'voice_arousal']):
        facial_ar = df['facial_arousal'].dropna()
        voice_ar = df['voice_arousal'].dropna()
        
        if len(facial_ar) > 0 and len(voice_ar) > 0:
            min_len = min(len(facial_ar), len(voice_ar))
            if min_len > 2:
                corr, _ = stats.spearmanr(facial_ar[:min_len], voice_ar[:min_len])
                stats_dict['arousal_modality_agreement'] = corr
    
    return stats_dict

def correlate_fusion_with_summary():
    """Main function to correlate fusion data with summary grades"""
    
    # Load summary grades
    summary_df = load_summary_grades()
    
    # Extract fusion stats for each participant
    fusion_data = []
    
    for _, row in summary_df.iterrows():
        group = row['Group']
        name = row['Name']
        
        fusion_stats = extract_fusion_stats(group, name)
        if fusion_stats:
            fusion_data.append(fusion_stats)
    
    # Create DataFrame
    fusion_df = pd.DataFrame(fusion_data)
    
    # Merge with summary grades
    merged_df = pd.merge(
        summary_df,
        fusion_df,
        left_on=['Group', 'Name'],
        right_on=['group', 'name'],
        how='inner'
    )
    
    # Drop duplicate columns
    merged_df = merged_df.drop(['group', 'name'], axis=1)
    
    # Save merged data
    merged_df.to_csv('fusion_summary_merged.csv', index=False)
    print(f"✅ Merged data saved to: fusion_summary_merged.csv")
    print(f"   {len(merged_df)} participants with complete data")
    print(f"   {len(merged_df.columns)} total features extracted")
    
    # Calculate correlations
    print("\n" + "="*70)
    print("COMPREHENSIVE CORRELATION ANALYSIS: Fusion Features vs Summary Grading")
    print("="*70)
    
    # Get all fusion-related columns
    fusion_metrics = []
    for col in merged_df.columns:
        if col not in ['Group', 'Name', 'Overall_Percentage', 'Semantic_Similarity', 
                      'Topic_Coverage', 'Factual_Accuracy', 'Letter_Grade',
                      'Original_Words', 'Summary_Words', 'Compression_Ratio',
                      'fused_dominant_emotion']:  # Exclude categorical
            if any(keyword in col for keyword in ['fused_', 'combined_', 'modality_']):
                fusion_metrics.append(col)
    
    # Summary grading metrics
    summary_metrics = [
        'Overall_Percentage',
        'Semantic_Similarity',
        'Topic_Coverage',
        'Factual_Accuracy'
    ]
    
    # Calculate correlations
    correlation_results = []
    
    for summary_metric in summary_metrics:
        print(f"\n{summary_metric}:")
        print("-" * 70)
        
        metric_correlations = []
        
        for fusion_metric in fusion_metrics:
            if fusion_metric in merged_df.columns:
                # Skip if all NaN or constant
                if merged_df[fusion_metric].isna().all() or merged_df[fusion_metric].std() == 0:
                    continue
                    
                # Calculate Spearman correlation
                valid_data = merged_df[[fusion_metric, summary_metric]].dropna()
                if len(valid_data) < 3:
                    continue
                    
                corr, p_value = stats.spearmanr(
                    valid_data[fusion_metric],
                    valid_data[summary_metric]
                )
                
                metric_correlations.append({
                    'metric': fusion_metric,
                    'corr': corr,
                    'p_value': p_value
                })
                
                correlation_results.append({
                    'Summary_Metric': summary_metric,
                    'Fusion_Metric': fusion_metric,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Abs_Correlation': abs(corr)
                })
        
        # Sort and show top correlations
        metric_correlations.sort(key=lambda x: abs(x['corr']), reverse=True)
        
        print("\n  TOP CORRELATIONS:")
        for i, item in enumerate(metric_correlations[:10]):
            sig_marker = "***" if item['p_value'] < 0.001 else "**" if item['p_value'] < 0.01 else "*" if item['p_value'] < 0.05 else ""
            print(f"  {i+1:2}. {item['metric']:40} | r = {item['corr']:7.3f} | p = {item['p_value']:.4f} {sig_marker}")
    
    # Save correlation results
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    corr_df.to_csv('fusion_summary_correlations.csv', index=False)
    print(f"\n✅ Correlation results saved to: fusion_summary_correlations.csv")
    print(f"   {len(corr_df)} total correlations calculated")
    print(f"   {len(corr_df[corr_df['Significant'] == 'Yes'])} significant correlations (p < 0.05)")
    
    # Show overall strongest correlations
    print("\n" + "="*70)
    print("TOP 15 STRONGEST CORRELATIONS (ACROSS ALL METRICS)")
    print("="*70)
    top_correlations = corr_df.head(15)
    for idx, row in top_correlations.iterrows():
        sig = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
        print(f"{row['Summary_Metric']:25} ~ {row['Fusion_Metric']:40} | r={row['Correlation']:7.3f} p={row['P_Value']:.4f} {sig}")
    
    # Create correlation heatmap
    create_correlation_heatmap(merged_df, fusion_metrics[:30], summary_metrics)
    
    # Group-level analysis
    print("\n" + "="*70)
    print("GROUP-LEVEL ANALYSIS")
    print("="*70)
    
    for group in ['neutral', 'opposing', 'similar']:
        group_data = merged_df[merged_df['Group'] == group]
        if len(group_data) > 0:
            print(f"\n{group.upper()} (n={len(group_data)}):")
            print(f"  Average Overall Score: {group_data['Overall_Percentage'].mean():.1f}%")
            
            if 'fused_valence_mean' in group_data.columns:
                print(f"  Average Fused Valence: {group_data['fused_valence_mean'].mean():.3f}")
            if 'fused_arousal_mean' in group_data.columns:
                print(f"  Average Fused Arousal: {group_data['fused_arousal_mean'].mean():.3f}")
            if 'fused_intensity_mean' in group_data.columns:
                print(f"  Average Fused Intensity: {group_data['fused_intensity_mean'].mean():.3f}")
            if 'fused_overall_volatility' in group_data.columns:
                print(f"  Average Fused Volatility: {group_data['fused_overall_volatility'].mean():.3f}")
    
    return merged_df, corr_df

def create_correlation_heatmap(merged_df, fusion_metrics, summary_metrics):
    """Create a correlation heatmap"""
    
    # Select only available metrics
    available_fusion = [m for m in fusion_metrics if m in merged_df.columns and merged_df[m].std() > 0]
    available_summary = [m for m in summary_metrics if m in merged_df.columns]
    
    if len(available_fusion) == 0 or len(available_summary) == 0:
        print("⚠️  Not enough valid metrics for heatmap")
        return
    
    # Calculate correlation matrix
    corr_matrix = merged_df[available_fusion + available_summary].corr()
    
    # Extract the cross-correlation
    cross_corr = corr_matrix.loc[available_fusion, available_summary]
    
    # Create heatmap
    plt.figure(figsize=(10, max(12, len(available_fusion) * 0.3)))
    sns.heatmap(
        cross_corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={'label': 'Spearman Correlation'},
        linewidths=0.5
    )
    
    plt.title('Fusion Features vs Summary Grading Correlations', fontsize=14, fontweight='bold')
    plt.xlabel('Summary Grading Metrics', fontsize=12)
    plt.ylabel('Fusion Feature Metrics', fontsize=12)
    plt.tight_layout()
    
    plt.savefig('fusion_summary_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap saved to: fusion_summary_correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    merged_df, corr_df = correlate_fusion_with_summary()
