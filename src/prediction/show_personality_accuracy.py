#!/usr/bin/env python3
"""
Generate simple bar chart visualization for personality-enhanced predictions
with accuracy summary
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# UTF-8 encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Load results
df = pd.read_csv('personality_enhanced_predictions.csv')

# Calculate accuracy metrics for combined model
predictions = df['Predicted_Combined'].values
actuals = df['Actual'].values
errors = predictions - actuals
abs_errors = np.abs(errors)

mae = np.mean(abs_errors)
rmse = np.sqrt(np.mean(errors**2))
median_ae = np.median(abs_errors)
r, p_value = stats.pearsonr(actuals, predictions)
r2 = r**2

# Create accuracy categories
df['Abs_Error_Combined'] = abs_errors
df['Accuracy_Category'] = df['Abs_Error_Combined'].apply(
    lambda x: 'Excellent' if x < 5 else 'Good' if x < 10 else 'Fair' if x < 15 else 'Poor'
)

excellent = (df['Abs_Error_Combined'] < 5).sum()
good = ((df['Abs_Error_Combined'] >= 5) & (df['Abs_Error_Combined'] < 10)).sum()
fair = ((df['Abs_Error_Combined'] >= 10) & (df['Abs_Error_Combined'] < 15)).sum()
poor = (df['Abs_Error_Combined'] >= 15).sum()

# Print accuracy summary
print('='*80)
print('PERSONALITY-ENHANCED MODEL ACCURACY SUMMARY')
print('='*80)
print('\nModel: Ridge Regression (Emotional + Personality Features)')
print('Cross-Validation: Leave-One-Out (LOOCV)')
print('Features: 15 emotional + 8 personality = 23 total')

print('\n--- Overall Performance ---')
print(f'Mean Absolute Error (MAE):  {mae:.2f}%')
print(f'Root Mean Squared Error:     {rmse:.2f}%')
print(f'Median Absolute Error:       {median_ae:.2f}%')
print(f'Max Error:                   {abs_errors.max():.2f}%')
print(f'Min Error:                   {abs_errors.min():.2f}%')

print('\n--- Correlation ---')
sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
print(f'Pearson r:  {r:.3f}{sig} (p = {p_value:.6f})')
print(f'R-squared:  {r2:.3f} (explains {r2*100:.1f}% of variance)')

print('\n--- Accuracy Breakdown ---')
print(f'Excellent (<5% error):   {excellent}/15 ({excellent/15*100:.1f}%)')
print(f'Good (5-10% error):      {good}/15 ({good/15*100:.1f}%)')
print(f'Fair (10-15% error):     {fair}/15 ({fair/15*100:.1f}%)')
print(f'Poor (>15% error):       {poor}/15 ({poor/15*100:.1f}%)')

print('\n--- Group Performance ---')
group_stats = df.groupby('Group').agg({
    'Abs_Error_Combined': ['mean', 'std', 'min', 'max']
})
print(group_stats.round(2))

print('\n--- Best Predictions (Top 5) ---')
best = df.nsmallest(5, 'Abs_Error_Combined')[['Name', 'Group', 'Actual', 'Predicted_Combined', 'Abs_Error_Combined']]
print(best.to_string(index=False))

print('\n--- Worst Predictions (Bottom 5) ---')
worst = df.nlargest(5, 'Abs_Error_Combined')[['Name', 'Group', 'Actual', 'Predicted_Combined', 'Abs_Error_Combined']]
print(worst.to_string(index=False))

print('\n' + '='*80)
print('KEY INSIGHTS:')
print('='*80)
print(f'✓ Model achieves {r:.3f} correlation (r² = {r2:.3f})')
print(f'✓ Average prediction error: ±{mae:.1f}%')
print(f'✓ {(excellent+good)/15*100:.0f}% of predictions within ±10% error')
best_group = group_stats.loc[group_stats[('Abs_Error_Combined', 'mean')].idxmin()]
worst_group = group_stats.loc[group_stats[('Abs_Error_Combined', 'mean')].idxmax()]
print(f'✓ Best performance: {group_stats[("Abs_Error_Combined", "mean")].min():.1f}% MAE ({group_stats[("Abs_Error_Combined", "mean")].idxmin().capitalize()} group)')
print(f'✓ Most challenging: {group_stats[("Abs_Error_Combined", "mean")].max():.1f}% MAE ({group_stats[("Abs_Error_Combined", "mean")].idxmax().capitalize()} group)')
print('✓ Personality traits enhance prediction by providing stable context')
print('='*80)

# Create visualization matching individual_predictions_simple style
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(df))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, df['Actual'], width, label='Actual Score',
               color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, df['Predicted_Combined'], width, label='Predicted Score',
               color='coral', alpha=0.8, edgecolor='black', linewidth=1.2)

# Customize plot
ax.set_xlabel('Participant', fontsize=13, fontweight='bold')
ax.set_ylabel('Summary Quality Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Personality-Enhanced Prediction Model: Actual vs Predicted Scores\n' +
             f'Ridge Regression (LOOCV) | R² = {r2:.3f}, r = {r:.3f}{sig}, MAE = {mae:.2f}%',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df['Name'], rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

# Add horizontal reference line at 50%
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Add value labels on bars (show only for bars that fit)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 5:  # Only show label if bar is tall enough
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

# Add text box with accuracy metrics
textstr = f'Performance Metrics:\n'
textstr += f'MAE: {mae:.2f}%\n'
textstr += f'RMSE: {rmse:.2f}%\n'
textstr += f'r: {r:.3f}{sig}\n'
textstr += f'R²: {r2:.3f}\n'
textstr += f'p-value: {p_value:.4f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=props, family='monospace')

plt.tight_layout()
plt.savefig('personality_enhanced_predictions_simple.png', dpi=300, bbox_inches='tight')
print(f'\n✓ Visualization saved to: personality_enhanced_predictions_simple.png')
