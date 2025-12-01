#!/usr/bin/env python3
"""
Compare original fusion model vs personality-enhanced model
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("MODEL COMPARISON: ORIGINAL FUSION vs PERSONALITY-ENHANCED")
print("="*80)

# Load both datasets
original_df = pd.read_csv('individual_predictions.csv')
enhanced_df = pd.read_csv('personality_enhanced_predictions.csv')

# Calculate metrics for original model
orig_predictions = original_df['Predicted'].values
orig_actuals = original_df['Actual'].values
orig_errors = orig_predictions - orig_actuals
orig_abs_errors = np.abs(orig_errors)

orig_mae = np.mean(orig_abs_errors)
orig_rmse = np.sqrt(np.mean(orig_errors**2))
orig_r, orig_p = stats.pearsonr(orig_actuals, orig_predictions)
orig_r2 = orig_r**2

# Calculate metrics for personality-enhanced model
enh_predictions = enhanced_df['Predicted_Combined'].values
enh_actuals = enhanced_df['Actual'].values
enh_errors = enh_predictions - enh_actuals
enh_abs_errors = np.abs(enh_errors)

enh_mae = np.mean(enh_abs_errors)
enh_rmse = np.sqrt(np.mean(enh_errors**2))
enh_r, enh_p = stats.pearsonr(enh_actuals, enh_predictions)
enh_r2 = enh_r**2

# Calculate improvements
mae_change = ((enh_mae - orig_mae) / orig_mae) * 100
rmse_change = ((enh_rmse - orig_rmse) / orig_rmse) * 100
r_change = ((enh_r - orig_r) / orig_r) * 100
r2_change = ((enh_r2 - orig_r2) / orig_r2) * 100

print("\n" + "="*80)
print("PERFORMANCE METRICS COMPARISON")
print("="*80)

print("\n┌─────────────────────────┬──────────────┬─────────────────┬─────────────┐")
print("│ Metric                  │ Original     │ Enhanced        │ Change      │")
print("├─────────────────────────┼──────────────┼─────────────────┼─────────────┤")
print(f"│ Features                │ 15 emotional │ 15 emo + 8 pers │ +8 traits   │")
print(f"│ Best Model              │ Ridge        │ Ridge           │ Same        │")
print("├─────────────────────────┼──────────────┼─────────────────┼─────────────┤")
print(f"│ Mean Absolute Error     │ {orig_mae:>11.2f}% │ {enh_mae:>14.2f}% │ {mae_change:>+10.1f}% │")
print(f"│ Root Mean Squared Error │ {orig_rmse:>11.2f}% │ {enh_rmse:>14.2f}% │ {rmse_change:>+10.1f}% │")
print(f"│ Pearson r               │ {orig_r:>12.3f} │ {enh_r:>15.3f} │ {r_change:>+10.1f}% │")
print(f"│ R² (variance explained) │ {orig_r2:>12.3f} │ {enh_r2:>15.3f} │ {r2_change:>+10.1f}% │")
orig_sig = '***' if orig_p < 0.001 else '**' if orig_p < 0.01 else '*' if orig_p < 0.05 else 'ns'
enh_sig = '***' if enh_p < 0.001 else '**' if enh_p < 0.01 else '*' if enh_p < 0.05 else 'ns'
print(f"│ Statistical Significance│ {orig_sig:>12} │ {enh_sig:>15} │             │")
print(f"│ p-value                 │ {orig_p:>12.6f} │ {enh_p:>15.6f} │             │")
print("└─────────────────────────┴──────────────┴─────────────────┴─────────────┘")

# Accuracy categories
orig_excellent = (orig_abs_errors < 5).sum()
orig_good = ((orig_abs_errors >= 5) & (orig_abs_errors < 10)).sum()
orig_fair = ((orig_abs_errors >= 10) & (orig_abs_errors < 15)).sum()
orig_poor = (orig_abs_errors >= 15).sum()

enh_excellent = (enh_abs_errors < 5).sum()
enh_good = ((enh_abs_errors >= 5) & (enh_abs_errors < 10)).sum()
enh_fair = ((enh_abs_errors >= 10) & (enh_abs_errors < 15)).sum()
enh_poor = (enh_abs_errors >= 15).sum()

print("\n" + "="*80)
print("ACCURACY DISTRIBUTION")
print("="*80)

print("\n┌─────────────────┬──────────────┬─────────────────┬──────────────┐")
print("│ Category        │ Original     │ Enhanced        │ Change       │")
print("├─────────────────┼──────────────┼─────────────────┼──────────────┤")
print(f"│ Excellent (<5%) │ {orig_excellent}/15 ({orig_excellent/15*100:>4.1f}%) │ {enh_excellent}/15 ({enh_excellent/15*100:>5.1f}%) │ {enh_excellent-orig_excellent:>+4} ({(enh_excellent-orig_excellent)/15*100:>+5.1f}%) │")
print(f"│ Good (5-10%)    │ {orig_good}/15 ({orig_good/15*100:>4.1f}%) │ {enh_good}/15 ({enh_good/15*100:>5.1f}%) │ {enh_good-orig_good:>+4} ({(enh_good-orig_good)/15*100:>+5.1f}%) │")
print(f"│ Fair (10-15%)   │ {orig_fair}/15 ({orig_fair/15*100:>4.1f}%) │ {enh_fair}/15 ({enh_fair/15*100:>5.1f}%) │ {enh_fair-orig_fair:>+4} ({(enh_fair-orig_fair)/15*100:>+5.1f}%) │")
print(f"│ Poor (>15%)     │ {orig_poor}/15 ({orig_poor/15*100:>4.1f}%) │ {enh_poor}/15 ({enh_poor/15*100:>5.1f}%) │ {enh_poor-orig_poor:>+4} ({(enh_poor-orig_poor)/15*100:>+5.1f}%) │")
print("└─────────────────┴──────────────┴─────────────────┴──────────────┘")

orig_success = orig_excellent + orig_good
enh_success = enh_excellent + enh_good
print(f"\nSuccess Rate (±10% error):")
print(f"  Original: {orig_success}/15 ({orig_success/15*100:.0f}%)")
print(f"  Enhanced: {enh_success}/15 ({enh_success/15*100:.0f}%)")
print(f"  Change: {enh_success-orig_success:+d} participants ({(enh_success-orig_success)/15*100:+.0f}%)")

# Group performance comparison
print("\n" + "="*80)
print("GROUP PERFORMANCE")
print("="*80)

original_df['Abs_Error'] = orig_abs_errors
enhanced_df['Abs_Error_Combined'] = enh_abs_errors

orig_groups = original_df.groupby('Group')['Abs_Error'].mean()
enh_groups = enhanced_df.groupby('Group')['Abs_Error_Combined'].mean()

print("\n┌──────────┬──────────────┬─────────────────┬──────────────┐")
print("│ Group    │ Original MAE │ Enhanced MAE    │ Change       │")
print("├──────────┼──────────────┼─────────────────┼──────────────┤")
for group in ['neutral', 'opposing', 'similar']:
    orig_val = orig_groups[group]
    enh_val = enh_groups[group]
    change = ((enh_val - orig_val) / orig_val) * 100
    print(f"│ {group.capitalize():<8} │ {orig_val:>11.2f}% │ {enh_val:>14.2f}% │ {change:>+11.1f}% │")
print("└──────────┴──────────────┴─────────────────┴──────────────┘")

# Individual-level comparison
print("\n" + "="*80)
print("INDIVIDUAL-LEVEL COMPARISON")
print("="*80)

# Match participants
merged = original_df.merge(enhanced_df[['Name', 'Predicted_Combined', 'Abs_Error_Combined']], 
                           on='Name', suffixes=('_orig', '_enh'))
merged['Error_Diff'] = merged['Abs_Error'] - merged['Abs_Error_Combined']
merged['Improved'] = merged['Error_Diff'] > 0

improved_count = merged['Improved'].sum()
worsened_count = (~merged['Improved']).sum()

print(f"\nIndividual Results:")
print(f"  Improved: {improved_count}/15 ({improved_count/15*100:.0f}%)")
print(f"  Worsened: {worsened_count}/15 ({worsened_count/15*100:.0f}%)")
print(f"  Average improvement: {merged['Error_Diff'].mean():.2f}% (negative = better)")

print("\nMost Improved:")
most_improved = merged.nlargest(5, 'Error_Diff')[['Name', 'Group', 'Abs_Error', 'Abs_Error_Combined', 'Error_Diff']]
most_improved.columns = ['Name', 'Group', 'Original Error', 'Enhanced Error', 'Improvement']
print(most_improved.to_string(index=False))

print("\nMost Worsened:")
most_worsened = merged.nsmallest(5, 'Error_Diff')[['Name', 'Group', 'Abs_Error', 'Abs_Error_Combined', 'Error_Diff']]
most_worsened.columns = ['Name', 'Group', 'Original Error', 'Enhanced Error', 'Change']
print(most_worsened.to_string(index=False))

# Statistical test
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TEST")
print("="*80)

# Paired t-test on absolute errors
from scipy.stats import ttest_rel, wilcoxon
t_stat, t_pval = ttest_rel(orig_abs_errors, enh_abs_errors)
w_stat, w_pval = wilcoxon(orig_abs_errors, enh_abs_errors)

print(f"\nPaired t-test (comparing absolute errors):")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {t_pval:.6f}")
print(f"  Result: {'Significant' if t_pval < 0.05 else 'Not significant'} at α=0.05")

print(f"\nWilcoxon signed-rank test (non-parametric):")
print(f"  statistic: {w_stat:.3f}")
print(f"  p-value: {w_pval:.6f}")
print(f"  Result: {'Significant' if w_pval < 0.05 else 'Not significant'} at α=0.05")

# Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if enh_r2 > orig_r2:
    print("\n✅ PERSONALITY ENHANCEMENT: IMPROVEMENT DETECTED")
    print(f"\nKey Improvements:")
    print(f"  • R² increased by {r2_change:+.1f}% (explains {(enh_r2-orig_r2)*100:.1f}% more variance)")
    print(f"  • MAE {'decreased' if mae_change < 0 else 'increased'} by {abs(mae_change):.1f}%")
    print(f"  • RMSE {'decreased' if rmse_change < 0 else 'increased'} by {abs(rmse_change):.1f}%")
    print(f"  • {improved_count} out of 15 participants showed improved predictions")
    
    if enh_p < 0.05 and orig_p >= 0.05:
        print(f"  • Achieved statistical significance (p={enh_p:.4f})")
    
    print(f"\nHOWEVER:")
    if mae_change > 0 or rmse_change > 0:
        print(f"  ⚠ Error metrics got worse - this suggests overfitting or model instability")
        print(f"  ⚠ The R² improvement may not translate to better practical predictions")
else:
    print("\n❌ PERSONALITY ENHANCEMENT: NO IMPROVEMENT")
    print(f"\nThe enhanced model performed {'worse' if enh_r2 < orig_r2 else 'the same'}:")
    print(f"  • R² {'decreased' if r2_change < 0 else 'unchanged'} by {abs(r2_change):.1f}%")
    print(f"  • MAE {'increased' if mae_change > 0 else 'decreased'} by {abs(mae_change):.1f}%")

print("\n" + "="*80)

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison: Original Fusion vs Personality-Enhanced', 
             fontsize=16, fontweight='bold')

# Plot 1: R² comparison
ax1 = axes[0, 0]
models = ['Original\n(Fusion Only)', 'Enhanced\n(Fusion + Personality)']
r2_values = [orig_r2, enh_r2]
colors = ['steelblue', 'coral']
bars = ax1.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Variance Explained (R²)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.grid(True, axis='y', alpha=0.3)
for bar, val in zip(bars, r2_values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
            f'{val:.3f}\n({val*100:.1f}%)',
            ha='center', fontsize=11, fontweight='bold')

# Plot 2: Error comparison
ax2 = axes[0, 1]
x = np.arange(2)
width = 0.35
bars1 = ax2.bar(x - width/2, [orig_mae, enh_mae], width, label='MAE',
               color='lightcoral', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x + width/2, [orig_rmse, enh_rmse], width, label='RMSE',
               color='lightblue', alpha=0.7, edgecolor='black')
ax2.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Errors', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.grid(True, axis='y', alpha=0.3)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 3: Accuracy distribution
ax3 = axes[1, 0]
categories = ['Excellent\n(<5%)', 'Good\n(5-10%)', 'Fair\n(10-15%)', 'Poor\n(>15%)']
orig_counts = [orig_excellent, orig_good, orig_fair, orig_poor]
enh_counts = [enh_excellent, enh_good, enh_fair, enh_poor]
x = np.arange(len(categories))
width = 0.35
bars1 = ax3.bar(x - width/2, orig_counts, width, label='Original',
               color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x + width/2, enh_counts, width, label='Enhanced',
               color='coral', alpha=0.7, edgecolor='black')
ax3.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
ax3.set_title('Accuracy Distribution', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(categories)
ax3.legend()
ax3.grid(True, axis='y', alpha=0.3)
ax3.set_ylim(0, 8)

# Plot 4: Individual improvements
ax4 = axes[1, 1]
improvement = merged.sort_values('Error_Diff', ascending=False)
colors = ['green' if x > 0 else 'red' for x in improvement['Error_Diff']]
bars = ax4.barh(improvement['Name'], improvement['Error_Diff'], 
                color=colors, alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax4.set_xlabel('Error Reduction (% points)', fontsize=12, fontweight='bold')
ax4.set_title('Individual-Level Changes\n(Positive = Improved)', 
              fontsize=13, fontweight='bold')
ax4.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_original_vs_enhanced.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparison visualization saved to: model_comparison_original_vs_enhanced.png")

