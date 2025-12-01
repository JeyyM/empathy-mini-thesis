#!/usr/bin/env python3
"""
Visualize individual predictions with personality enhancement
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
df = pd.read_csv('personality_enhanced_predictions.csv')

# Create detailed comparison visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('Personality-Enhanced Prediction: Individual Results', 
             fontsize=18, fontweight='bold', y=0.98)

# Plot 1: Bar chart comparison (all three models)
ax1 = fig.add_subplot(gs[0, :])
x = np.arange(len(df))
width = 0.2

bars1 = ax1.bar(x - width, df['Actual'], width, label='Actual', 
                color='black', alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x, df['Predicted_Emotional'], width, 
                label='Emotional Only', color='blue', alpha=0.6, edgecolor='black')
bars3 = ax1.bar(x + width, df['Predicted_Combined'], width,
                label='Combined', color='purple', alpha=0.6, edgecolor='black')

ax1.set_xlabel('Participant', fontsize=12, fontweight='bold')
ax1.set_ylabel('Summary Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Actual vs Predicted Scores: Emotional vs Combined Model', 
              fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Name'], rotation=45, ha='right')
ax1.legend(fontsize=10)
ax1.grid(True, axis='y', alpha=0.3)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Plot 2: Error comparison (emotional only)
ax2 = fig.add_subplot(gs[1, 0])
colors_emotional = ['green' if abs(err) < 10 else 'orange' if abs(err) < 15 else 'red' 
                    for err in df['Error_Emotional']]
bars = ax2.barh(df['Name'], df['Error_Emotional'], color=colors_emotional, 
                alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Prediction Error (%)', fontsize=11, fontweight='bold')
ax2.set_title('Emotional Features Only\nPrediction Errors', 
              fontsize=12, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)
mae_emotional = df['Error_Emotional'].abs().mean()
ax2.text(0.98, 0.02, f'MAE = {mae_emotional:.1f}%', 
         transform=ax2.transAxes, fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 3: Error comparison (combined)
ax3 = fig.add_subplot(gs[1, 1])
colors_combined = ['green' if abs(err) < 10 else 'orange' if abs(err) < 15 else 'red' 
                   for err in df['Error_Combined']]
bars = ax3.barh(df['Name'], df['Error_Combined'], color=colors_combined,
                alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Prediction Error (%)', fontsize=11, fontweight='bold')
ax3.set_title('Combined (Emotional + Personality)\nPrediction Errors',
              fontsize=12, fontweight='bold')
ax3.grid(True, axis='x', alpha=0.3)
mae_combined = df['Error_Combined'].abs().mean()
ax3.text(0.98, 0.02, f'MAE = {mae_combined:.1f}%',
         transform=ax3.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Error reduction analysis
ax4 = fig.add_subplot(gs[2, 0])
error_reduction = df['Error_Emotional'].abs() - df['Error_Combined'].abs()
colors_reduction = ['green' if x > 0 else 'red' for x in error_reduction]
bars = ax4.barh(df['Name'], error_reduction, color=colors_reduction,
                alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Error Reduction (%)', fontsize=11, fontweight='bold')
ax4.set_title('Improvement from Adding Personality\n(Positive = Better)',
              fontsize=12, fontweight='bold')
ax4.grid(True, axis='x', alpha=0.3)
improved = (error_reduction > 0).sum()
total = len(df)
ax4.text(0.98, 0.02, f'{improved}/{total} improved ({improved/total*100:.0f}%)',
         transform=ax4.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Plot 5: Group performance
ax5 = fig.add_subplot(gs[2, 1])
group_stats = df.groupby('Group').agg({
    'Error_Emotional': lambda x: x.abs().mean(),
    'Error_Combined': lambda x: x.abs().mean()
})
x_groups = np.arange(len(group_stats))
width = 0.35
bars1 = ax5.bar(x_groups - width/2, group_stats['Error_Emotional'], width,
                label='Emotional Only', color='blue', alpha=0.6, edgecolor='black')
bars2 = ax5.bar(x_groups + width/2, group_stats['Error_Combined'], width,
                label='Combined', color='purple', alpha=0.6, edgecolor='black')

ax5.set_xlabel('Group', fontsize=11, fontweight='bold')
ax5.set_ylabel('Mean Absolute Error (%)', fontsize=11, fontweight='bold')
ax5.set_title('Group-Level Performance Comparison',
              fontsize=12, fontweight='bold')
ax5.set_xticks(x_groups)
ax5.set_xticklabels(group_stats.index.str.capitalize())
ax5.legend()
ax5.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.savefig('personality_enhancement_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Detailed visualization saved to: personality_enhancement_detailed.png")

# Create summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nOverall Performance:")
print(f"  Emotional Only - MAE: {mae_emotional:.2f}%")
print(f"  Combined       - MAE: {mae_combined:.2f}%")
print(f"  Improvement:     {mae_emotional - mae_combined:.2f}% reduction")

print("\nParticipants Improved:")
print(f"  {improved}/{total} ({improved/total*100:.0f}%)")

print("\nBest Predictions (Combined Model):")
df['Abs_Error_Combined'] = df['Error_Combined'].abs()
best_5 = df.nsmallest(5, 'Abs_Error_Combined')[['Name', 'Actual', 'Predicted_Combined', 'Error_Combined']]
print(best_5.to_string(index=False))

print("\nWorst Predictions (Combined Model):")
worst_5 = df.nlargest(5, 'Abs_Error_Combined')[['Name', 'Actual', 'Predicted_Combined', 'Error_Combined']]
print(worst_5.to_string(index=False))

print("\nGroup Performance (MAE):")
print(group_stats.round(2).to_string())

print("\n" + "="*80)
