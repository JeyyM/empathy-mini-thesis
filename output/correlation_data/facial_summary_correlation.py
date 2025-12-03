#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlate facial emotion data with summary grading results
"""

import os
import sys
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

def extract_facial_stats(group, name):
    """Extract comprehensive facial emotion statistics from the emotion data CSV"""
    # Construct the file path
    base_path = f"../results/{group.capitalize()}/{name}"
    csv_file = f"Final{name}_ml_emotion_data.csv"
    file_path = os.path.join(base_path, csv_file)
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return None
    
    # Load the emotion data
    df = pd.read_csv(file_path)
    
    # Extract facial emotion columns
    facial_cols = {
        'valence': 'facial_valence',
        'arousal': 'facial_arousal',
        'intensity': 'facial_intensity',
        'excitement': 'facial_excitement',
        'calmness': 'facial_calmness',
        'positivity': 'facial_positivity',
        'negativity': 'facial_negativity',
        'happy': 'facial_happy',
        'angry': 'facial_angry',
        'sad': 'facial_sad',
        'fear': 'facial_fear',
        'surprise': 'facial_surprise',
        'disgust': 'facial_disgust',
        'neutral': 'facial_neutral'
    }
    
    stats_dict = {
        'group': group,
        'name': name
    }
    
    # Basic statistics for each facial metric
    for metric_name, col_name in facial_cols.items():
        if col_name in df.columns:
            data = df[col_name]
            stats_dict[f'{metric_name}_mean'] = data.mean()
            stats_dict[f'{metric_name}_std'] = data.std()
            stats_dict[f'{metric_name}_min'] = data.min()
            stats_dict[f'{metric_name}_max'] = data.max()
            stats_dict[f'{metric_name}_median'] = data.median()
            
            # VOLATILITY METRICS
            stats_dict[f'{metric_name}_range'] = data.max() - data.min()
            stats_dict[f'{metric_name}_variance'] = data.var()
            stats_dict[f'{metric_name}_cv'] = data.std() / abs(data.mean()) if data.mean() != 0 else 0  # Coefficient of variation
            
            # CHANGE METRICS - How much the emotion changes frame-to-frame
            if len(data) > 1:
                changes = data.diff().abs()
                stats_dict[f'{metric_name}_mean_change'] = changes.mean()
                stats_dict[f'{metric_name}_max_change'] = changes.max()
                stats_dict[f'{metric_name}_total_change'] = changes.sum()
            
            # TREND METRICS - Is the emotion increasing or decreasing over time?
            if len(data) > 2:
                x = np.arange(len(data))
                slope, intercept = np.polyfit(x, data, 1)
                stats_dict[f'{metric_name}_trend_slope'] = slope
                stats_dict[f'{metric_name}_trend_direction'] = 1 if slope > 0 else -1 if slope < 0 else 0
    
    # EMOTIONAL STATE TIME ANALYSIS
    if 'facial_quadrant' in df.columns:
        quadrant_counts = df['facial_quadrant'].value_counts()
        total_frames = len(df)
        
        for quadrant in ['Happy', 'Calm', 'Stressed', 'Tired']:
            count = quadrant_counts.get(quadrant, 0)
            stats_dict[f'time_in_{quadrant.lower()}'] = count
            stats_dict[f'pct_time_{quadrant.lower()}'] = (count / total_frames) * 100 if total_frames > 0 else 0
        
        # TRANSITION FREQUENCY - How often does emotional state change?
        if len(df) > 1:
            transitions = (df['facial_quadrant'] != df['facial_quadrant'].shift()).sum()
            stats_dict['quadrant_transitions'] = transitions
            stats_dict['transition_rate'] = transitions / len(df) if len(df) > 0 else 0
    
    # DOMINANT EMOTION ANALYSIS
    emotion_cols = ['facial_happy', 'facial_angry', 'facial_sad', 'facial_fear', 
                   'facial_surprise', 'facial_disgust', 'facial_neutral']
    
    if all(col in df.columns for col in emotion_cols):
        # Find which emotion is dominant at each frame
        emotion_means = {col.replace('facial_', ''): df[col].mean() for col in emotion_cols}
        stats_dict['dominant_emotion'] = max(emotion_means, key=emotion_means.get)
        stats_dict['dominant_emotion_strength'] = max(emotion_means.values())
        
        # Calculate emotional diversity (entropy-like measure)
        emotion_values = [df[col].mean() for col in emotion_cols]
        total = sum(emotion_values)
        if total > 0:
            proportions = [v/total for v in emotion_values]
            diversity = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
            stats_dict['emotional_diversity'] = diversity
    
    # OVERALL EMOTIONAL STABILITY
    core_emotions = ['facial_valence', 'facial_arousal', 'facial_intensity']
    if all(col in df.columns for col in core_emotions):
        combined_std = sum(df[col].std() for col in core_emotions) / len(core_emotions)
        stats_dict['overall_emotional_stability'] = 1 / (1 + combined_std)  # Higher = more stable
        stats_dict['overall_emotional_volatility'] = combined_std
    
    # PEAK EMOTIONAL MOMENTS
    if 'facial_intensity' in df.columns:
        intensity_data = df['facial_intensity']
        top_10_pct = int(len(intensity_data) * 0.1)
        if top_10_pct > 0:
            top_intensities = intensity_data.nlargest(top_10_pct)
            stats_dict['peak_intensity_mean'] = top_intensities.mean()
            stats_dict['peak_intensity_max'] = top_intensities.max()
    
    return stats_dict

def correlate_facial_with_summary():
    """Main function to correlate facial data with summary grades"""
    
    # Load summary grades
    summary_df = load_summary_grades()
    
    # Extract facial stats for each participant
    facial_data = []
    
    for _, row in summary_df.iterrows():
        group = row['Group']
        name = row['Name']
        
        facial_stats = extract_facial_stats(group, name)
        if facial_stats:
            facial_data.append(facial_stats)
    
    # Create DataFrame
    facial_df = pd.DataFrame(facial_data)
    
    # Merge with summary grades
    merged_df = pd.merge(
        summary_df,
        facial_df,
        left_on=['Group', 'Name'],
        right_on=['group', 'name'],
        how='inner'
    )
    
    # Drop duplicate columns
    merged_df = merged_df.drop(['group', 'name'], axis=1)
    
    # Save merged data
    merged_df.to_csv('facial_summary_merged.csv', index=False)
    print(f"✅ Merged data saved to: facial_summary_merged.csv")
    print(f"   {len(merged_df)} participants with complete data")
    print(f"   {len(merged_df.columns)} total features extracted")
    
    # Calculate correlations
    print("\n" + "="*70)
    print("COMPREHENSIVE CORRELATION ANALYSIS: Facial Emotions vs Summary Grading")
    print("="*70)
    
    # Primary facial metrics to analyze - NOW INCLUDING VOLATILITY AND TEMPORAL FEATURES
    facial_metrics = []
    
    # Get all facial-related columns (excluding group/name and summary metrics)
    for col in merged_df.columns:
        if col not in ['Group', 'Name', 'Overall_Percentage', 'Semantic_Similarity', 
                      'Topic_Coverage', 'Factual_Accuracy', 'Letter_Grade',
                      'Original_Words', 'Summary_Words', 'Compression_Ratio',
                      'dominant_emotion']:  # Exclude categorical
            if any(keyword in col for keyword in ['valence', 'arousal', 'intensity', 'excitement', 
                                                  'calmness', 'positivity', 'negativity', 
                                                  'happy', 'angry', 'sad', 'fear', 'surprise', 
                                                  'disgust', 'neutral', 'time_in', 'pct_time',
                                                  'transition', 'diversity', 'stability', 
                                                  'volatility', 'peak', 'trend', 'change', 'range']):
                facial_metrics.append(col)
    
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
        
        for facial_metric in facial_metrics:
            if facial_metric in merged_df.columns:
                # Skip if all NaN or constant
                if merged_df[facial_metric].isna().all() or merged_df[facial_metric].std() == 0:
                    continue
                    
                # Calculate Spearman correlation
                valid_data = merged_df[[facial_metric, summary_metric]].dropna()
                if len(valid_data) < 3:  # Need at least 3 points
                    continue
                    
                corr, p_value = stats.spearmanr(
                    valid_data[facial_metric],
                    valid_data[summary_metric]
                )
                
                metric_correlations.append({
                    'metric': facial_metric,
                    'corr': corr,
                    'p_value': p_value
                })
                
                correlation_results.append({
                    'Summary_Metric': summary_metric,
                    'Facial_Metric': facial_metric,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Abs_Correlation': abs(corr)
                })
        
        # Sort by absolute correlation and show top correlations
        metric_correlations.sort(key=lambda x: abs(x['corr']), reverse=True)
        
        print("\n  TOP CORRELATIONS:")
        for i, item in enumerate(metric_correlations[:10]):  # Show top 10
            sig_marker = "***" if item['p_value'] < 0.001 else "**" if item['p_value'] < 0.01 else "*" if item['p_value'] < 0.05 else ""
            print(f"  {i+1:2}. {item['metric']:35} | r = {item['corr']:7.3f} | p = {item['p_value']:.4f} {sig_marker}")
    
    # Save correlation results
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    corr_df.to_csv('facial_summary_correlations.csv', index=False)
    print(f"\n✅ Correlation results saved to: facial_summary_correlations.csv")
    print(f"   {len(corr_df)} total correlations calculated")
    print(f"   {len(corr_df[corr_df['Significant'] == 'Yes'])} significant correlations (p < 0.05)")
    
    # Show overall strongest correlations
    print("\n" + "="*70)
    print("TOP 15 STRONGEST CORRELATIONS (ACROSS ALL METRICS)")
    print("="*70)
    top_correlations = corr_df.head(15)
    for idx, row in top_correlations.iterrows():
        sig = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
        print(f"{row['Summary_Metric']:25} ~ {row['Facial_Metric']:35} | r={row['Correlation']:7.3f} p={row['P_Value']:.4f} {sig}")
    
    # Create correlation heatmap
    create_correlation_heatmap(merged_df, facial_metrics, summary_metrics)
    
    # Group-level analysis
    print("\n" + "="*70)
    print("GROUP-LEVEL ANALYSIS")
    print("="*70)
    
    for group in ['neutral', 'opposing', 'similar']:
        group_data = merged_df[merged_df['Group'] == group]
        print(f"\n{group.upper()} (n={len(group_data)}):")
        print(f"  Average Overall Score: {group_data['Overall_Percentage'].mean():.1f}%")
        print(f"  Average Valence: {group_data['valence_mean'].mean():.3f}")
        print(f"  Average Arousal: {group_data['arousal_mean'].mean():.3f}")
        print(f"  Average Intensity: {group_data['intensity_mean'].mean():.3f}")
        print(f"  Average Positivity: {group_data['positivity_mean'].mean():.3f}")
        print(f"  Average Negativity: {group_data['negativity_mean'].mean():.3f}")
    
    return merged_df, corr_df

def create_correlation_heatmap(merged_df, facial_metrics, summary_metrics):
    """Create a correlation heatmap"""
    
    # Select only available metrics
    available_facial = [m for m in facial_metrics if m in merged_df.columns]
    available_summary = [m for m in summary_metrics if m in merged_df.columns]
    
    # Calculate correlation matrix
    corr_matrix = merged_df[available_facial + available_summary].corr()
    
    # Extract the cross-correlation between facial and summary metrics
    cross_corr = corr_matrix.loc[available_facial, available_summary]
    
    # Create heatmap
    plt.figure(figsize=(10, 12))
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
    
    plt.title('Facial Emotions vs Summary Grading Correlations', fontsize=14, fontweight='bold')
    plt.xlabel('Summary Grading Metrics', fontsize=12)
    plt.ylabel('Facial Emotion Metrics', fontsize=12)
    plt.tight_layout()
    
    plt.savefig('facial_summary_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap saved to: facial_summary_correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    merged_df, corr_df = correlate_facial_with_summary()
