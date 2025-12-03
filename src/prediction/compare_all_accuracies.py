#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare accuracy analysis across all prediction models
"""

import pandas as pd
import numpy as np

def analyze_accuracy(predictions_file, model_name):
    """Analyze prediction accuracy for a given model"""
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} - PREDICTION ACCURACY ANALYSIS")
    print(f"{'='*80}")
    
    df = pd.read_csv(predictions_file)
    
    # Calculate error ranges
    excellent = df[df['Abs_Error'] <= 5]
    good = df[(df['Abs_Error'] > 5) & (df['Abs_Error'] <= 10)]
    acceptable = df[(df['Abs_Error'] > 10) & (df['Abs_Error'] <= 15)]
    poor = df[df['Abs_Error'] > 15]
    
    total = len(df)
    
    print(f"\nTotal Predictions: {total}")
    print(f"\nError Distribution:")
    print(f"  Excellent (≤5%):      {len(excellent):2d} ({len(excellent)/total*100:5.1f}%) {list(excellent['Name'])}")
    print(f"  Good (5-10%):         {len(good):2d} ({len(good)/total*100:5.1f}%) {list(good['Name'])}")
    print(f"  Acceptable (10-15%):  {len(acceptable):2d} ({len(acceptable)/total*100:5.1f}%) {list(acceptable['Name'])}")
    print(f"  Poor (>15%):          {len(poor):2d} ({len(poor)/total*100:5.1f}%) {list(poor['Name'])}")
    
    success_rate = (len(excellent) + len(good)) / total * 100
    print(f"\n✅ Success Rate (≤10% error): {success_rate:.1f}%")
    
    print(f"\nPerformance Statistics:")
    print(f"  Mean Absolute Error:  {df['Abs_Error'].mean():.2f}%")
    print(f"  Median Error:         {df['Abs_Error'].median():.2f}%")
    print(f"  Std Dev:              {df['Abs_Error'].std():.2f}%")
    print(f"  Min Error:            {df['Abs_Error'].min():.2f}%")
    print(f"  Max Error:            {df['Abs_Error'].max():.2f}%")
    
    # Best and worst predictions
    best = df.loc[df['Abs_Error'].idxmin()]
    worst = df.loc[df['Abs_Error'].idxmax()]
    
    print(f"\n🏆 Best Prediction:")
    print(f"  {best['Name']}: Actual={best['Actual']:.1f}%, Predicted={best['Predicted']:.1f}%, Error={best['Abs_Error']:.2f}%")
    
    print(f"\n❌ Worst Prediction:")
    print(f"  {worst['Name']}: Actual={worst['Actual']:.1f}%, Predicted={worst['Predicted']:.1f}%, Error={worst['Abs_Error']:.2f}%")
    
    return {
        'Model': model_name,
        'Total': total,
        'Excellent': len(excellent),
        'Good': len(good),
        'Acceptable': len(acceptable),
        'Poor': len(poor),
        'Success_Rate': success_rate,
        'MAE': df['Abs_Error'].mean(),
        'Median_Error': df['Abs_Error'].median(),
        'Std_Dev': df['Abs_Error'].std(),
        'Min_Error': df['Abs_Error'].min(),
        'Max_Error': df['Abs_Error'].max()
    }

# Analyze all models
results = []

# Original fusion model (from predict_summary_scores.py)
results.append(analyze_accuracy('individual_predictions.csv', 'Original Fusion Model'))

# Facial-only model
results.append(analyze_accuracy('facial_only_predictions.csv', 'Facial-Only Model'))

# Voice-only model
results.append(analyze_accuracy('voice_only_predictions.csv', 'Voice-Only Model'))

# Fusion-only model (rerun with isolated features)
results.append(analyze_accuracy('fusion_only_predictions.csv', 'Fusion-Only Model'))

# Create comparison summary
print(f"\n{'='*80}")
print("COMPREHENSIVE COMPARISON SUMMARY")
print(f"{'='*80}")

summary_df = pd.DataFrame(results)
print("\n" + summary_df.to_string(index=False))

# Save comparison
summary_df.to_csv('all_models_accuracy_comparison.csv', index=False)
print(f"\n✅ Comparison saved to: all_models_accuracy_comparison.csv")

# Quick ranking
print(f"\n{'='*80}")
print("RANKING BY SUCCESS RATE (≤10% error)")
print(f"{'='*80}")
ranking = summary_df.sort_values('Success_Rate', ascending=False)
for i, row in ranking.iterrows():
    print(f"{i+1}. {row['Model']:25s} - {row['Success_Rate']:5.1f}% success, MAE={row['MAE']:.2f}%")

print(f"\n{'='*80}")
print("✅ ACCURACY ANALYSIS COMPLETE!")
print(f"{'='*80}")
