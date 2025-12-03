#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize correlations between emotion features and summary scores using scatter plots.
Groups features by category and shows all three modalities (facial, voice, fusion) together.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

OUTPUTS_DIR = Path("outputs") / "correlation_analysis"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_path(filename: str) -> Path:
    """Return the first existing path for the given filename across common roots."""
    candidates = [
        OUTPUTS_DIR / filename,
        Path("output") / "correlation_data" / filename,
        Path("outputs") / "correlation_data" / filename,
        Path("data") / "correlation_data" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    # Default to OUTPUTS_DIR/filename even if missing (will likely be written later)
    return OUTPUTS_DIR / filename

def load_all_data():
    """Load merged data from all three modalities"""
    facial_path = _resolve_path('facial_summary_merged.csv')
    voice_path = _resolve_path('voice_summary_merged.csv')
    fusion_path = _resolve_path('fusion_summary_merged.csv')
    if not facial_path.exists() or not voice_path.exists() or not fusion_path.exists():
        print("⚠️  One or more merged CSVs are missing. Checked:")
        print(f"    - {facial_path}")
        print(f"    - {voice_path}")
        print(f"    - {fusion_path}")
    facial_df = pd.read_csv(facial_path)
    voice_df = pd.read_csv(voice_path)
    fusion_df = pd.read_csv(fusion_path)
    
    return facial_df, voice_df, fusion_df

def create_scatter_with_stats(ax, x_data, y_data, title, xlabel, color, modality_name):
    """Create a scatter plot with regression line and correlation stats"""
    # Remove NaN values
    mask = ~(pd.isna(x_data) | pd.isna(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    if len(x_clean) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                transform=ax.transAxes, fontsize=10)
        ax.set_title(f"{modality_name}\n{title}", fontsize=10, fontweight='bold')
        return
    
    # Calculate Spearman correlation
    rho, p_value = stats.spearmanr(x_clean, y_clean)
    
    # Create scatter plot
    ax.scatter(x_clean, y_clean, alpha=0.6, s=80, color=color, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(x_clean, y_clean, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
    ax.plot(x_line, p(x_line), color=color, linestyle='--', linewidth=2, alpha=0.8)
    
    # Add correlation stats
    sig_marker = ''
    if p_value < 0.001:
        sig_marker = '***'
    elif p_value < 0.01:
        sig_marker = '**'
    elif p_value < 0.05:
        sig_marker = '*'
    
    stats_text = f'ρ = {rho:.3f}{sig_marker}\np = {p_value:.4f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel('Overall Summary Score (%)', fontsize=9)
    ax.set_title(f"{modality_name}\n{title}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

def visualize_core_dimensions(facial_df, voice_df, fusion_df):
    """Visualize core emotional dimensions: Valence, Arousal, Intensity"""
    print("\n📊 Generating Core Dimensions scatter plots...")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Core Emotional Dimensions vs Summary Score', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Define the features to plot
    features = [
        ('valence', 'Valence'),
        ('arousal', 'Arousal'),
        ('intensity', 'Intensity')
    ]
    
    colors = {
        'facial': '#FF6B6B',
        'voice': '#4ECDC4',
        'fusion': '#95E1D3'
    }
    
    for idx, (feature, label) in enumerate(features):
        # Facial - use _mean suffix
        ax1 = plt.subplot(3, 3, idx*3 + 1)
        facial_col = f'{feature}_mean'
        if facial_col in facial_df.columns:
            create_scatter_with_stats(
                ax1, facial_df[facial_col], facial_df['Overall_Percentage'],
                label, f'Facial {label} (mean)', colors['facial'], 'Facial'
            )
        
        # Voice - use _mean suffix
        ax2 = plt.subplot(3, 3, idx*3 + 2)
        voice_col = f'{feature}_mean'
        if voice_col in voice_df.columns:
            create_scatter_with_stats(
                ax2, voice_df[voice_col], voice_df['Overall_Percentage'],
                label, f'Voice {label} (mean)', colors['voice'], 'Voice'
            )
        
        # Fusion - use _mean suffix
        ax3 = plt.subplot(3, 3, idx*3 + 3)
        fusion_col = f'{feature}_mean'
        if fusion_col in fusion_df.columns:
            create_scatter_with_stats(
                ax3, fusion_df[fusion_col], fusion_df['Overall_Percentage'],
                label, f'Fusion {label} (mean)', colors['fusion'], 'Fusion'
            )
    
    plt.tight_layout()
    out_path = OUTPUTS_DIR / 'scatter_core_dimensions.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {out_path.resolve()}")
    plt.close()

def visualize_emotions(facial_df, voice_df, fusion_df):
    """Visualize individual emotions"""
    print("\n📊 Generating Emotions scatter plots...")
    
    # Define emotions
    emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
    
    colors = {
        'facial': '#FF6B6B',
        'voice': '#4ECDC4',
        'fusion': '#95E1D3'
    }
    
    # Create multiple figures for emotions (3x3 grid, so 3 emotions per page)
    emotions_per_page = 3
    num_pages = (len(emotions) + emotions_per_page - 1) // emotions_per_page
    
    for page in range(num_pages):
        fig = plt.figure(figsize=(18, 12))
        start_idx = page * emotions_per_page
        end_idx = min(start_idx + emotions_per_page, len(emotions))
        page_emotions = emotions[start_idx:end_idx]
        
        fig.suptitle(f'Emotion Correlations vs Summary Score (Page {page+1}/{num_pages})', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for idx, emotion in enumerate(page_emotions):
            # Facial - use _mean suffix
            ax1 = plt.subplot(3, 3, idx*3 + 1)
            facial_col = f'{emotion}_mean'
            if facial_col in facial_df.columns:
                create_scatter_with_stats(
                    ax1, facial_df[facial_col], facial_df['Overall_Percentage'],
                    emotion.capitalize(), f'Facial {emotion.capitalize()} (mean)', 
                    colors['facial'], 'Facial'
                )
            
            # Voice - use _mean suffix
            ax2 = plt.subplot(3, 3, idx*3 + 2)
            voice_col = f'{emotion}_mean'
            if voice_col in voice_df.columns:
                create_scatter_with_stats(
                    ax2, voice_df[voice_col], voice_df['Overall_Percentage'],
                    emotion.capitalize(), f'Voice {emotion.capitalize()} (mean)', 
                    colors['voice'], 'Voice'
                )
            
            # Fusion - use _mean suffix
            ax3 = plt.subplot(3, 3, idx*3 + 3)
            fusion_col = f'{emotion}_mean'
            if fusion_col in fusion_df.columns:
                create_scatter_with_stats(
                    ax3, fusion_df[fusion_col], fusion_df['Overall_Percentage'],
                    emotion.capitalize(), f'Fusion {emotion.capitalize()} (mean)', 
                    colors['fusion'], 'Fusion'
                )
        
        plt.tight_layout()
        filename = OUTPUTS_DIR / f'scatter_emotions_page{page+1}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filename.resolve()}")
        plt.close()

def visualize_volatility_features(facial_df, voice_df, fusion_df):
    """Visualize volatility and dynamic features"""
    print("\n📊 Generating Volatility & Dynamics scatter plots...")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Emotional Volatility & Dynamics vs Summary Score', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = {
        'facial': '#FF6B6B',
        'voice': '#4ECDC4',
        'fusion': '#95E1D3'
    }
    
    # Define features to plot - using the ones that exist across modalities
    features = [
        ('valence_std', 'Valence Volatility (SD)'),
        ('arousal_std', 'Arousal Volatility (SD)'),
        ('intensity_std', 'Intensity Volatility (SD)')
    ]
    
    for idx, (feature, label) in enumerate(features):
        # Facial - direct column name
        ax1 = plt.subplot(3, 3, idx*3 + 1)
        if feature in facial_df.columns:
            create_scatter_with_stats(
                ax1, facial_df[feature], facial_df['Overall_Percentage'],
                label, label, colors['facial'], 'Facial'
            )
        
        # Voice - direct column name
        ax2 = plt.subplot(3, 3, idx*3 + 2)
        if feature in voice_df.columns:
            create_scatter_with_stats(
                ax2, voice_df[feature], voice_df['Overall_Percentage'],
                label, label, colors['voice'], 'Voice'
            )
        
        # Fusion - direct column name
        ax3 = plt.subplot(3, 3, idx*3 + 3)
        if feature in fusion_df.columns:
            create_scatter_with_stats(
                ax3, fusion_df[feature], fusion_df['Overall_Percentage'],
                label, label, colors['fusion'], 'Fusion'
            )
    
    plt.tight_layout()
    out_path = OUTPUTS_DIR / 'scatter_volatility.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {out_path.resolve()}")
    plt.close()

def visualize_transitions(facial_df, voice_df, fusion_df):
    """Visualize transition features (the strongest predictors)"""
    print("\n📊 Generating Transitions & Changes scatter plots...")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Emotional Transitions & Changes vs Summary Score', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = {
        'facial': '#FF6B6B',
        'voice': '#4ECDC4',
        'fusion': '#95E1D3'
    }
    
    # Define features to plot
    features = [
        ('valence_total_change', 'Valence Total Change'),
        ('arousal_total_change', 'Arousal Total Change'),
        ('quadrant_transitions', 'Quadrant Transitions')
    ]
    
    for idx, (feature, label) in enumerate(features):
        # Facial - direct column name
        ax1 = plt.subplot(3, 3, idx*3 + 1)
        if feature in facial_df.columns:
            create_scatter_with_stats(
                ax1, facial_df[feature], facial_df['Overall_Percentage'],
                label, label, colors['facial'], 'Facial'
            )
        
        # Voice - direct column name
        ax2 = plt.subplot(3, 3, idx*3 + 2)
        if feature in voice_df.columns:
            create_scatter_with_stats(
                ax2, voice_df[feature], voice_df['Overall_Percentage'],
                label, label, colors['voice'], 'Voice'
            )
        
        # Fusion - direct column name
        ax3 = plt.subplot(3, 3, idx*3 + 3)
        if feature in fusion_df.columns:
            create_scatter_with_stats(
                ax3, fusion_df[feature], fusion_df['Overall_Percentage'],
                label, label, colors['fusion'], 'Fusion'
            )
    
    plt.tight_layout()
    out_path = OUTPUTS_DIR / 'scatter_transitions.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {out_path.resolve()}")
    plt.close()

def visualize_voice_acoustic(voice_df):
    """Visualize voice-specific acoustic features"""
    print("\n📊 Generating Voice Acoustic Features scatter plots...")
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Voice Acoustic Features vs Summary Score', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    color = '#4ECDC4'
    
    # Define acoustic features
    features = [
        ('pitch_mean_mean', 'Mean Pitch (Hz)'),
        ('volume_mean_mean', 'Mean Volume'),
        ('speech_rate_mean', 'Speech Rate'),
        ('pitch_std_mean', 'Pitch Variability (SD)'),
        ('volume_std_mean', 'Volume Variability (SD)'),
        ('tremor_mean', 'Voice Tremor'),
        ('pitch_mean_total_change', 'Pitch Total Change'),
        ('volume_mean_total_change', 'Volume Total Change'),
        ('volume_std_trend_direction', 'Volume Trend Direction')
    ]
    
    for idx, (feature, label) in enumerate(features):
        ax = plt.subplot(3, 3, idx + 1)
        if feature in voice_df.columns:
            create_scatter_with_stats(
                ax, voice_df[feature], voice_df['Overall_Percentage'],
                label, label, color, 'Voice'
            )
    
    plt.tight_layout()
    out_path = OUTPUTS_DIR / 'scatter_voice_acoustic.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {out_path.resolve()}")
    plt.close()

def create_summary_table(facial_df, voice_df, fusion_df):
    """Create a summary table of the strongest correlations"""
    print("\n📊 Generating summary statistics table...")
    
    summary_data = []
    
    # Core dimensions - use _mean suffix for facial, voice prefix for voice
    for feature in ['valence', 'arousal', 'intensity']:
        # Facial - uses suffix pattern like valence_mean
        facial_col = f'{feature}_mean'
        if facial_col in facial_df.columns:
            mask = ~(pd.isna(facial_df[facial_col]) | pd.isna(facial_df['Overall_Percentage']))
            if mask.sum() >= 3:
                rho, p = stats.spearmanr(facial_df[facial_col][mask], 
                                        facial_df['Overall_Percentage'][mask])
                summary_data.append({
                    'Category': 'Core Dimension',
                    'Feature': f'{feature.capitalize()} (mean)',
                    'Modality': 'Facial',
                    'Correlation_rho': rho,
                    'P_Value': p,
                    'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                })
        
        # Voice - uses prefix pattern like voice_valence
        voice_col = f'{feature}'
        if voice_col in voice_df.columns:
            mask = ~(pd.isna(voice_df[voice_col]) | pd.isna(voice_df['Overall_Percentage']))
            if mask.sum() >= 3:
                rho, p = stats.spearmanr(voice_df[voice_col][mask], 
                                        voice_df['Overall_Percentage'][mask])
                summary_data.append({
                    'Category': 'Core Dimension',
                    'Feature': feature.capitalize(),
                    'Modality': 'Voice',
                    'Correlation_rho': rho,
                    'P_Value': p,
                    'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                })
        
        # Fusion - check multiple patterns
        for prefix in ['combined_', 'fused_', '']:
            fusion_col = f'{prefix}{feature}'
            if fusion_col in fusion_df.columns:
                mask = ~(pd.isna(fusion_df[fusion_col]) | pd.isna(fusion_df['Overall_Percentage']))
                if mask.sum() >= 3:
                    rho, p = stats.spearmanr(fusion_df[fusion_col][mask], 
                                            fusion_df['Overall_Percentage'][mask])
                    summary_data.append({
                        'Category': 'Core Dimension',
                        'Feature': f'{feature.capitalize()} ({prefix if prefix else "base"})',
                        'Modality': 'Fusion',
                        'Correlation_rho': rho,
                        'P_Value': p,
                        'Significant': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    })
                break  # Only use first matching pattern
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) == 0:
        print("⚠️  No correlation data generated")
        return
    
    summary_df = summary_df.sort_values('Correlation_rho', key=abs, ascending=False)
    out_csv = OUTPUTS_DIR / 'scatter_correlations_summary.csv'
    summary_df.to_csv(out_csv, index=False)
    
    print("\n" + "="*80)
    print("TOP CORRELATIONS SUMMARY")
    print("="*80)
    print(summary_df.head(15).to_string(index=False))
    print(f"\n✅ Saved: scatter_correlations_summary.csv")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("COMPREHENSIVE CORRELATION VISUALIZATION: SCATTER PLOTS")
    print("="*80)
    print("Analyzing: Facial, Voice, and Fusion modalities")
    print("Target: Overall Summary Score (%)")
    print("Method: Spearman correlation with scatter plots")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data from all modalities...")
    facial_df, voice_df, fusion_df = load_all_data()
    print(f"   Facial: {len(facial_df)} participants, {len(facial_df.columns)} features")
    print(f"   Voice: {len(voice_df)} participants, {len(voice_df.columns)} features")
    print(f"   Fusion: {len(fusion_df)} participants, {len(fusion_df.columns)} features")
    
    # Create visualizations
    visualize_core_dimensions(facial_df, voice_df, fusion_df)
    visualize_emotions(facial_df, voice_df, fusion_df)
    visualize_volatility_features(facial_df, voice_df, fusion_df)
    visualize_transitions(facial_df, voice_df, fusion_df)
    visualize_voice_acoustic(voice_df)
    create_summary_table(facial_df, voice_df, fusion_df)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  📊 {OUTPUTS_DIR / 'scatter_core_dimensions.png'} - Valence, Arousal, Intensity")
    print(f"  📊 {OUTPUTS_DIR / 'scatter_emotions_page1.png'} - Happy, Sad, Angry")
    print(f"  📊 {OUTPUTS_DIR / 'scatter_emotions_page2.png'} - Fear, Surprise, Disgust")
    print(f"  📊 {OUTPUTS_DIR / 'scatter_emotions_page3.png'} - Neutral")
    print(f"  📊 {OUTPUTS_DIR / 'scatter_volatility.png'} - Emotional volatility metrics")
    print(f"  📊 {OUTPUTS_DIR / 'scatter_transitions.png'} - Quadrant transitions & changes")
    print(f"  📊 {OUTPUTS_DIR / 'scatter_voice_acoustic.png'} - Voice acoustic features")
    print(f"  📄 {OUTPUTS_DIR / 'scatter_correlations_summary.csv'} - Summary statistics")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
