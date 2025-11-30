#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show mean accuracies for each modality overall and by group
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("MEAN ACCURACIES BY MODALITY AND GROUP")
print("="*80)

# Load all prediction files
facial_df = pd.read_csv('facial_only_predictions.csv')
voice_df = pd.read_csv('voice_only_predictions.csv')
fusion_df = pd.read_csv('fusion_only_predictions.csv')

# Function to calculate group statistics
def calculate_group_stats(df, modality_name):
    print(f"\n{'='*80}")
    print(f"{modality_name.upper()} MODALITY")
    print(f"{'='*80}")
    
    # Overall statistics
    overall_mae = df['Abs_Error'].mean()
    overall_rmse = np.sqrt((df['Error']**2).mean())
    
    print(f"\n📊 Overall Performance:")
    print(f"   Mean Absolute Error:  {overall_mae:.2f}%")
    print(f"   RMSE:                 {overall_rmse:.2f}%")
    
    # Group statistics
    print(f"\n📊 By Group:")
    print(f"{'Group':<12} {'Count':>6} {'Mean MAE':>10} {'Mean RMSE':>11} {'Mean Actual':>12} {'Mean Predicted':>15}")
    print("-"*80)
    
    group_stats = []
    for group in ['neutral', 'opposing', 'similar']:
        group_data = df[df['Group'] == group]
        if len(group_data) > 0:
            mae = group_data['Abs_Error'].mean()
            rmse = np.sqrt((group_data['Error']**2).mean())
            mean_actual = group_data['Actual'].mean()
            mean_predicted = group_data['Predicted'].mean()
            
            print(f"{group.capitalize():<12} {len(group_data):>6} {mae:>10.2f}% {rmse:>10.2f}% {mean_actual:>11.1f}% {mean_predicted:>14.1f}%")
            
            group_stats.append({
                'Modality': modality_name,
                'Group': group,
                'Count': len(group_data),
                'MAE': mae,
                'RMSE': rmse,
                'Mean_Actual': mean_actual,
                'Mean_Predicted': mean_predicted
            })
    
    return overall_mae, overall_rmse, group_stats

# Calculate for each modality
facial_mae, facial_rmse, facial_groups = calculate_group_stats(facial_df, 'Facial')
voice_mae, voice_rmse, voice_groups = calculate_group_stats(voice_df, 'Voice')
fusion_mae, fusion_rmse, fusion_groups = calculate_group_stats(fusion_df, 'Fusion')

# Overall comparison
print("\n" + "="*80)
print("OVERALL COMPARISON")
print("="*80)
print(f"\n{'Modality':<15} {'Mean MAE':>10} {'Mean RMSE':>11}")
print("-"*40)
print(f"{'Facial':<15} {facial_mae:>10.2f}% {facial_rmse:>10.2f}%")
print(f"{'Voice':<15} {voice_mae:>10.2f}% {voice_rmse:>10.2f}%")
print(f"{'Fusion':<15} {fusion_mae:>10.2f}% {fusion_rmse:>10.2f}%")

# Group comparison across modalities
print("\n" + "="*80)
print("GROUP PERFORMANCE COMPARISON ACROSS MODALITIES")
print("="*80)

all_group_stats = facial_groups + voice_groups + fusion_groups
group_df = pd.DataFrame(all_group_stats)

for group in ['neutral', 'opposing', 'similar']:
    print(f"\n📊 {group.upper()} Group:")
    print(f"{'Modality':<15} {'MAE':>10} {'RMSE':>11} {'Mean Actual':>12} {'Mean Predicted':>15}")
    print("-"*65)
    
    group_subset = group_df[group_df['Group'] == group]
    for _, row in group_subset.iterrows():
        print(f"{row['Modality']:<15} {row['MAE']:>10.2f}% {row['RMSE']:>10.2f}% {row['Mean_Actual']:>11.1f}% {row['Mean_Predicted']:>14.1f}%")

# Save to CSV
group_df.to_csv('modality_group_accuracies.csv', index=False)
print(f"\n✅ Group accuracies saved to: modality_group_accuracies.csv")

# Summary findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Find best modality overall
best_overall = min([
    ('Facial', facial_mae),
    ('Voice', voice_mae),
    ('Fusion', fusion_mae)
], key=lambda x: x[1])

print(f"\n🏆 Best Overall Modality: {best_overall[0]} (MAE = {best_overall[1]:.2f}%)")

# Find best modality for each group
print(f"\n🏆 Best Modality by Group:")
for group in ['neutral', 'opposing', 'similar']:
    group_subset = group_df[group_df['Group'] == group]
    best_group = group_subset.loc[group_subset['MAE'].idxmin()]
    print(f"   {group.capitalize()}: {best_group['Modality']} (MAE = {best_group['MAE']:.2f}%)")

# Prediction bias analysis
print(f"\n📈 Prediction Bias (Mean Predicted - Mean Actual):")
for group in ['neutral', 'opposing', 'similar']:
    print(f"\n   {group.capitalize()}:")
    group_subset = group_df[group_df['Group'] == group]
    for _, row in group_subset.iterrows():
        bias = row['Mean_Predicted'] - row['Mean_Actual']
        direction = "overpredicting" if bias > 0 else "underpredicting"
        print(f"      {row['Modality']:<10}: {bias:+6.1f}% ({direction})")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80 + "\n")
