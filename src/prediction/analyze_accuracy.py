#!/usr/bin/env python3
"""Quick analysis of prediction model accuracy"""

import pandas as pd
from scipy import stats

# Load results
df = pd.read_csv('output\prediction_analysis\individual_predictions.csv')

print('='*80)
print('PREDICTION MODEL ACCURACY SUMMARY')
print('='*80)
print('\nBest Model: Ridge Regression (Fusion Data)')
print('Cross-Validation: Leave-One-Out (LOOCV)')
print('Features: 15 emotional features (quadrant transitions, valence/arousal changes)')

print('\n--- Overall Performance ---')
print(f'Mean Absolute Error (MAE):  {df["Abs_Error"].mean():.2f}%')
print(f'Root Mean Squared Error:     {((df["Error"]**2).mean())**0.5:.2f}%')
print(f'Median Absolute Error:       {df["Abs_Error"].median():.2f}%')
print(f'Max Error:                   {df["Abs_Error"].max():.2f}%')
print(f'Min Error:                   {df["Abs_Error"].min():.2f}%')

print('\n--- Correlation ---')
r, p = stats.pearsonr(df['Actual'], df['Predicted'])
r2 = r**2
print(f'Pearson r:  {r:.3f} (p = {p:.6f})')
print(f'R-squared:  {r2:.3f} (explains {r2*100:.1f}% of variance)')

print('\n--- Accuracy Breakdown ---')
excellent = (df['Abs_Error'] < 5).sum()
good = ((df['Abs_Error'] >= 5) & (df['Abs_Error'] < 10)).sum()
fair = ((df['Abs_Error'] >= 10) & (df['Abs_Error'] < 15)).sum()
poor = (df['Abs_Error'] >= 15).sum()

print(f'Excellent (<5% error):   {excellent}/15 ({excellent/15*100:.1f}%)')
print(f'Good (5-10% error):      {good}/15 ({good/15*100:.1f}%)')
print(f'Fair (10-15% error):     {fair}/15 ({fair/15*100:.1f}%)')
print(f'Poor (>15% error):       {poor}/15 ({poor/15*100:.1f}%)')

print('\n--- Group Performance ---')
group_stats = df.groupby('Group').agg({
    'Abs_Error': ['mean', 'std', 'min', 'max'],
    'Error': 'mean'
})
print(group_stats.round(2))

print('\n--- Best Predictions (Top 5) ---')
best = df.nsmallest(5, 'Abs_Error')[['Name', 'Group', 'Actual', 'Predicted', 'Abs_Error']]
print(best.to_string(index=False))

print('\n--- Worst Predictions (Bottom 5) ---')
worst = df.nlargest(5, 'Abs_Error')[['Name', 'Group', 'Actual', 'Predicted', 'Abs_Error']]
print(worst.to_string(index=False))

print('\n' + '='*80)
print('KEY INSIGHTS:')
print('='*80)
print(f'✓ Model achieves {r:.3f} correlation (r² = {r2:.3f})')
print(f'✓ Average prediction error: ±{df["Abs_Error"].mean():.1f}%')
print(f'✓ {(excellent+good)/15*100:.0f}% of predictions within ±10% error')
print(f'✓ Best performance: {group_stats.loc[group_stats[("Abs_Error", "mean")].idxmin()]["Abs_Error"]["mean"]:.1f}% MAE ({group_stats[("Abs_Error", "mean")].idxmin().capitalize()} group)')
print(f'✓ Most challenging: {group_stats.loc[group_stats[("Abs_Error", "mean")].idxmax()]["Abs_Error"]["mean"]:.1f}% MAE ({group_stats[("Abs_Error", "mean")].idxmax().capitalize()} group)')
print('='*80)
