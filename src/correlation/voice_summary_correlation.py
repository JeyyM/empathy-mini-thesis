#!/usr/bin/env python3
"""
Correlate voice emotion data with summary grading results
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

def extract_voice_stats(group, name):
    """Extract comprehensive voice emotion statistics from the emotion data CSV"""
    # Construct the file path
    base_path = f"../results/{group.capitalize()}/{name}"
    csv_file = f"Final{name}_ml_emotion_data.csv"
    file_path = os.path.join(base_path, csv_file)
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return None
    
    # Load the emotion data
    df = pd.read_csv(file_path)
    
    # Extract voice emotion columns
    voice_cols = {
        # Core dimensions
        'arousal': 'voice_arousal',
        'valence': 'voice_valence',
        'dominance': 'voice_dominance',
        'intensity': 'voice_intensity',
        'stress': 'voice_stress',
        
        # Emotions
        'happy': 'voice_happy',
        'angry': 'voice_angry',
        'sad': 'voice_sad',
        'fear': 'voice_fear',
        'surprise': 'voice_surprise',
        'disgust': 'voice_disgust',
        'neutral': 'voice_neutral',
        
        # Acoustic features
        'pitch_mean': 'voice_pitch_mean',
        'pitch_std': 'voice_pitch_std',
        'pitch_range': 'voice_pitch_range',
        'pitch_variation': 'voice_pitch_variation',
        'volume_mean': 'voice_volume_mean',
        'volume_std': 'voice_volume_std',
        'volume_range': 'voice_volume_range',
        'spectral_centroid': 'voice_spectral_centroid',
        'spectral_bandwidth': 'voice_spectral_bandwidth',
        'spectral_rolloff': 'voice_spectral_rolloff',
        'zero_crossing_rate': 'voice_zero_crossing_rate',
        'speech_rate': 'voice_speech_rate',
        'silence_ratio': 'voice_silence_ratio',
        'harmonic_ratio': 'voice_harmonic_ratio',
        'tremor': 'voice_tremor'
    }
    
    stats_dict = {
        'group': group,
        'name': name
    }
    
    # Basic statistics for each voice metric
    for metric_name, col_name in voice_cols.items():
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
            stats_dict[f'{metric_name}_cv'] = data.std() / abs(data.mean()) if data.mean() != 0 else 0  # Coefficient of variation
            
            # CHANGE METRICS - How much the metric changes segment-to-segment
            if len(data) > 1:
                changes = data.diff().abs()
                stats_dict[f'{metric_name}_mean_change'] = changes.mean()
                stats_dict[f'{metric_name}_max_change'] = changes.max()
                stats_dict[f'{metric_name}_total_change'] = changes.sum()
            
            # TREND METRICS - Is the metric increasing or decreasing over time?
            if len(data) > 2:
                x = np.arange(len(data))
                try:
                    slope, intercept = np.polyfit(x, data, 1)
                    stats_dict[f'{metric_name}_trend_slope'] = slope
                    stats_dict[f'{metric_name}_trend_direction'] = 1 if slope > 0 else -1 if slope < 0 else 0
                except:
                    pass
    
    # DOMINANT EMOTION ANALYSIS
    emotion_cols = ['voice_happy', 'voice_angry', 'voice_sad', 'voice_fear', 
                   'voice_surprise', 'voice_disgust', 'voice_neutral']
    
    available_emotion_cols = [col for col in emotion_cols if col in df.columns]
    
    if len(available_emotion_cols) > 0:
        # Find which emotion is dominant
        emotion_means = {}
        for col in available_emotion_cols:
            data = df[col].dropna()
            if len(data) > 0:
                emotion_means[col.replace('voice_', '')] = data.mean()
        
        if emotion_means:
            stats_dict['dominant_emotion'] = max(emotion_means, key=emotion_means.get)
            stats_dict['dominant_emotion_strength'] = max(emotion_means.values())
            
            # Calculate emotional diversity (entropy-like measure)
            emotion_values = list(emotion_means.values())
            total = sum(emotion_values)
            if total > 0:
                proportions = [v/total for v in emotion_values]
                diversity = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
                stats_dict['emotional_diversity'] = diversity
    
    # OVERALL VOCAL STABILITY
    core_voice = ['voice_arousal', 'voice_valence', 'voice_intensity']
    available_core = [col for col in core_voice if col in df.columns]
    
    if len(available_core) > 0:
        stds = []
        for col in available_core:
            data = df[col].dropna()
            if len(data) > 0:
                stds.append(data.std())
        
        if stds:
            combined_std = sum(stds) / len(stds)
            stats_dict['overall_vocal_stability'] = 1 / (1 + combined_std)  # Higher = more stable
            stats_dict['overall_vocal_volatility'] = combined_std
    
    # PEAK EMOTIONAL MOMENTS
    if 'voice_intensity' in df.columns:
        intensity_data = df['voice_intensity'].dropna()
        if len(intensity_data) > 0:
            top_10_pct = max(1, int(len(intensity_data) * 0.1))
            top_intensities = intensity_data.nlargest(top_10_pct)
            stats_dict['peak_intensity_mean'] = top_intensities.mean()
            stats_dict['peak_intensity_max'] = top_intensities.max()
    
    # SPEECH PATTERN ANALYSIS
    if 'voice_speech_rate' in df.columns:
        speech_data = df['voice_speech_rate'].dropna()
        if len(speech_data) > 0:
            stats_dict['speech_rate_consistency'] = 1 / (1 + speech_data.std())
    
    if 'voice_silence_ratio' in df.columns:
        silence_data = df['voice_silence_ratio'].dropna()
        if len(silence_data) > 0:
            stats_dict['avg_silence_ratio'] = silence_data.mean()
            stats_dict['silence_variability'] = silence_data.std()
    
    # PITCH DYNAMICS
    if 'voice_pitch_mean' in df.columns:
        pitch_data = df['voice_pitch_mean'].dropna()
        if len(pitch_data) > 1:
            # Pitch modulation - how much pitch varies
            stats_dict['pitch_modulation'] = pitch_data.std() / pitch_data.mean() if pitch_data.mean() != 0 else 0
            
            # Pitch transitions
            pitch_changes = pitch_data.diff().abs()
            stats_dict['pitch_transitions'] = (pitch_changes > pitch_data.std()).sum()
    
    # ENERGY/VOLUME DYNAMICS
    if 'voice_volume_mean' in df.columns:
        volume_data = df['voice_volume_mean'].dropna()
        if len(volume_data) > 1:
            stats_dict['volume_modulation'] = volume_data.std() / volume_data.mean() if volume_data.mean() != 0 else 0
            
            # Volume transitions (significant changes)
            volume_changes = volume_data.diff().abs()
            stats_dict['volume_transitions'] = (volume_changes > volume_data.std()).sum()
    
    return stats_dict

def correlate_voice_with_summary():
    """Main function to correlate voice data with summary grades"""
    
    # Load summary grades
    summary_df = load_summary_grades()
    
    # Extract voice stats for each participant
    voice_data = []
    
    for _, row in summary_df.iterrows():
        group = row['Group']
        name = row['Name']
        
        voice_stats = extract_voice_stats(group, name)
        if voice_stats:
            voice_data.append(voice_stats)
    
    # Create DataFrame
    voice_df = pd.DataFrame(voice_data)
    
    # Merge with summary grades
    merged_df = pd.merge(
        summary_df,
        voice_df,
        left_on=['Group', 'Name'],
        right_on=['group', 'name'],
        how='inner'
    )
    
    # Drop duplicate columns
    merged_df = merged_df.drop(['group', 'name'], axis=1)
    
    # Save merged data
    merged_df.to_csv('voice_summary_merged.csv', index=False)
    print(f"✅ Merged data saved to: voice_summary_merged.csv")
    print(f"   {len(merged_df)} participants with complete data")
    print(f"   {len(merged_df.columns)} total features extracted")
    
    # Calculate correlations
    print("\n" + "="*70)
    print("COMPREHENSIVE CORRELATION ANALYSIS: Voice Features vs Summary Grading")
    print("="*70)
    
    # Get all voice-related columns
    voice_metrics = []
    for col in merged_df.columns:
        if col not in ['Group', 'Name', 'Overall_Percentage', 'Semantic_Similarity', 
                      'Topic_Coverage', 'Factual_Accuracy', 'Letter_Grade',
                      'Original_Words', 'Summary_Words', 'Compression_Ratio',
                      'dominant_emotion']:  # Exclude categorical
            if any(keyword in col for keyword in ['arousal', 'valence', 'dominance', 'intensity', 
                                                  'stress', 'happy', 'angry', 'sad', 'fear', 
                                                  'surprise', 'disgust', 'neutral', 'pitch', 
                                                  'volume', 'spectral', 'speech', 'silence',
                                                  'harmonic', 'tremor', 'diversity', 'stability', 
                                                  'volatility', 'peak', 'trend', 'change', 'range',
                                                  'modulation', 'transition', 'consistency']):
                voice_metrics.append(col)
    
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
        
        for voice_metric in voice_metrics:
            if voice_metric in merged_df.columns:
                # Skip if all NaN or constant
                if merged_df[voice_metric].isna().all() or merged_df[voice_metric].std() == 0:
                    continue
                    
                # Calculate Spearman correlation
                valid_data = merged_df[[voice_metric, summary_metric]].dropna()
                if len(valid_data) < 3:  # Need at least 3 points
                    continue
                    
                corr, p_value = stats.spearmanr(
                    valid_data[voice_metric],
                    valid_data[summary_metric]
                )
                
                metric_correlations.append({
                    'metric': voice_metric,
                    'corr': corr,
                    'p_value': p_value
                })
                
                correlation_results.append({
                    'Summary_Metric': summary_metric,
                    'Voice_Metric': voice_metric,
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
            print(f"  {i+1:2}. {item['metric']:40} | r = {item['corr']:7.3f} | p = {item['p_value']:.4f} {sig_marker}")
    
    # Save correlation results
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    corr_df.to_csv('voice_summary_correlations.csv', index=False)
    print(f"\n✅ Correlation results saved to: voice_summary_correlations.csv")
    print(f"   {len(corr_df)} total correlations calculated")
    print(f"   {len(corr_df[corr_df['Significant'] == 'Yes'])} significant correlations (p < 0.05)")
    
    # Show overall strongest correlations
    print("\n" + "="*70)
    print("TOP 15 STRONGEST CORRELATIONS (ACROSS ALL METRICS)")
    print("="*70)
    top_correlations = corr_df.head(15)
    for idx, row in top_correlations.iterrows():
        sig = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
        print(f"{row['Summary_Metric']:25} ~ {row['Voice_Metric']:40} | r={row['Correlation']:7.3f} p={row['P_Value']:.4f} {sig}")
    
    # Create correlation heatmap
    create_correlation_heatmap(merged_df, voice_metrics[:30], summary_metrics)  # Top 30 for readability
    
    # Group-level analysis
    print("\n" + "="*70)
    print("GROUP-LEVEL ANALYSIS")
    print("="*70)
    
    for group in ['neutral', 'opposing', 'similar']:
        group_data = merged_df[merged_df['Group'] == group]
        if len(group_data) > 0:
            print(f"\n{group.upper()} (n={len(group_data)}):")
            print(f"  Average Overall Score: {group_data['Overall_Percentage'].mean():.1f}%")
            
            if 'valence_mean' in group_data.columns:
                print(f"  Average Valence: {group_data['valence_mean'].mean():.3f}")
            if 'arousal_mean' in group_data.columns:
                print(f"  Average Arousal: {group_data['arousal_mean'].mean():.3f}")
            if 'intensity_mean' in group_data.columns:
                print(f"  Average Intensity: {group_data['intensity_mean'].mean():.3f}")
            if 'pitch_mean_mean' in group_data.columns:
                print(f"  Average Pitch: {group_data['pitch_mean_mean'].mean():.1f} Hz")
            if 'speech_rate_mean' in group_data.columns:
                print(f"  Average Speech Rate: {group_data['speech_rate_mean'].mean():.1f}")
    
    return merged_df, corr_df

def create_correlation_heatmap(merged_df, voice_metrics, summary_metrics):
    """Create a correlation heatmap"""
    
    # Select only available metrics
    available_voice = [m for m in voice_metrics if m in merged_df.columns and merged_df[m].std() > 0]
    available_summary = [m for m in summary_metrics if m in merged_df.columns]
    
    if len(available_voice) == 0 or len(available_summary) == 0:
        print("⚠️  Not enough valid metrics for heatmap")
        return
    
    # Calculate correlation matrix
    corr_matrix = merged_df[available_voice + available_summary].corr()
    
    # Extract the cross-correlation between voice and summary metrics
    cross_corr = corr_matrix.loc[available_voice, available_summary]
    
    # Create heatmap
    plt.figure(figsize=(10, max(12, len(available_voice) * 0.3)))
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
    
    plt.title('Voice Features vs Summary Grading Correlations', fontsize=14, fontweight='bold')
    plt.xlabel('Summary Grading Metrics', fontsize=12)
    plt.ylabel('Voice Feature Metrics', fontsize=12)
    plt.tight_layout()
    
    plt.savefig('voice_summary_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap saved to: voice_summary_correlation_heatmap.png")
    plt.close()

if __name__ == "__main__":
    merged_df, corr_df = correlate_voice_with_summary()
