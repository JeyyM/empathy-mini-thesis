#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Facial vs Voice vs Fusion Prediction Models
Runs all three modality-specific models and compares performance
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("MODALITY COMPARISON: FACIAL vs VOICE vs FUSION")
print("="*80)
print("Running three separate prediction models in isolation...")
print("="*80)

# Run each model
print("\n[1/3] Running FACIAL-only model...")
subprocess.run(["python", "predict_facial_only.py"], check=True)

print("\n[2/3] Running VOICE-only model...")
subprocess.run(["python", "predict_voice_only.py"], check=True)

print("\n[3/3] Running FUSION-only model...")
subprocess.run(["python", "predict_fusion_only.py"], check=True)

# Load results
facial_df = pd.read_csv('facial_only_predictions.csv')
voice_df = pd.read_csv('voice_only_predictions.csv')
fusion_df = pd.read_csv('fusion_only_predictions.csv')

# Calculate metrics
from scipy import stats

def calculate_metrics(df):
    mae = df['Abs_Error'].mean()
    rmse = (df['Error']**2).mean()**0.5
    r, p = stats.pearsonr(df['Actual'], df['Predicted'])
    r2 = r**2
    return {'MAE': mae, 'RMSE': rmse, 'r': r, 'p': p, 'R2': r2}

facial_metrics = calculate_metrics(facial_df)
voice_metrics = calculate_metrics(voice_df)
fusion_metrics = calculate_metrics(fusion_df)

# Create comparison table
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Modality': ['Facial', 'Voice', 'Fusion'],
    'MAE': [facial_metrics['MAE'], voice_metrics['MAE'], fusion_metrics['MAE']],
    'RMSE': [facial_metrics['RMSE'], voice_metrics['RMSE'], fusion_metrics['RMSE']],
    'r': [facial_metrics['r'], voice_metrics['r'], fusion_metrics['r']],
    'p-value': [facial_metrics['p'], voice_metrics['p'], fusion_metrics['p']],
    'R²': [facial_metrics['R2'], voice_metrics['R2'], fusion_metrics['R2']]
})

comparison_df = comparison_df.sort_values('R²', ascending=False)
print("\n" + comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('modality_comparison_results.csv', index=False)
print(f"\n✅ Comparison saved to: modality_comparison_results.csv")

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# R² comparison
ax = axes[0, 0]
colors = ['#3498db', '#e74c3c', '#9b59b6']
bars = ax.bar(comparison_df['Modality'], comparison_df['R²'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('R² (Variance Explained)', fontsize=11, fontweight='bold')
ax.set_title('Model Performance: R²', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(comparison_df['R²']) * 1.2)
for i, (bar, val) in enumerate(zip(bars, comparison_df['R²'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# MAE comparison
ax = axes[0, 1]
bars = ax.bar(comparison_df['Modality'], comparison_df['MAE'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Mean Absolute Error (%)', fontsize=11, fontweight='bold')
ax.set_title('Prediction Error: MAE', fontsize=12, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, comparison_df['MAE'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Correlation comparison
ax = axes[1, 0]
bars = ax.bar(comparison_df['Modality'], comparison_df['r'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Pearson r (Correlation)', fontsize=11, fontweight='bold')
ax.set_title('Correlation Strength', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(comparison_df['r']) * 1.2)
for i, (bar, val) in enumerate(zip(bars, comparison_df['r'])):
    sig = '***' if comparison_df['p-value'].iloc[i] < 0.001 else '**' if comparison_df['p-value'].iloc[i] < 0.01 else '*' if comparison_df['p-value'].iloc[i] < 0.05 else 'ns'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.3f}{sig}', ha='center', va='bottom', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Scatter comparison
ax = axes[1, 1]
ax.scatter(facial_df['Actual'], facial_df['Predicted'], alpha=0.6, s=80, label='Facial', color='#3498db', edgecolors='black', linewidth=0.5)
ax.scatter(voice_df['Actual'], voice_df['Predicted'], alpha=0.6, s=80, label='Voice', color='#e74c3c', edgecolors='black', linewidth=0.5)
ax.scatter(fusion_df['Actual'], fusion_df['Predicted'], alpha=0.6, s=80, label='Fusion', color='#9b59b6', edgecolors='black', linewidth=0.5)
min_val = min(facial_df['Actual'].min(), voice_df['Actual'].min(), fusion_df['Actual'].min())
max_val = max(facial_df['Actual'].max(), voice_df['Actual'].max(), fusion_df['Actual'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect')
ax.set_xlabel('Actual Score (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Predicted Score (%)', fontsize=11, fontweight='bold')
ax.set_title('Actual vs Predicted (All Modalities)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modality_comparison_visual.png', dpi=300, bbox_inches='tight')
print(f"✅ Comparison visualization saved to: modality_comparison_visual.png")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
best_modality = comparison_df.iloc[0]['Modality']
best_r2 = comparison_df.iloc[0]['R²']
best_mae = comparison_df.iloc[0]['MAE']

print(f"\n🏆 BEST MODALITY: {best_modality}")
print(f"   R² = {best_r2:.3f} (explains {best_r2*100:.1f}% of variance)")
print(f"   MAE = {best_mae:.2f}%")
print(f"   Correlation r = {comparison_df.iloc[0]['r']:.3f}")

print("\n" + "="*80)
print("✅ MODALITY COMPARISON COMPLETE!")
print("="*80)
