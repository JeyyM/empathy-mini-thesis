#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize individual predictions from the best model
Shows actual vs predicted scores for each participant
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def load_data():
    """Load fusion data"""
    df = pd.read_csv('../correlation/fusion_summary_merged.csv')
    return df

def get_top_features():
    """Return the top features used in the best model"""
    return [
        'combined_quadrant_transitions',
        'fused_quadrant_transitions',
        'disgust_total_change',
        'combined_valence_total_change',
        'angry_total_change',
        'combined_arousal_total_change',
        'excitement_trend_direction',
        'fear_min',
        'stress_total_change',
        'valence_modality_agreement',
        'happy_total_change',
        'negativity_total_change',
        'arousal_total_change',
        'valence_total_change',
        'positivity_total_change'
    ]

def run_loocv_prediction(df, features):
    """Run LOOCV and return predictions for each person"""
    X = df[features].values
    y = df['Overall_Percentage'].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # LOOCV
    loo = LeaveOneOut()
    scaler = StandardScaler()
    model = Ridge(alpha=1.0)
    
    predictions = []
    actuals = []
    names = []
    groups = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        predictions.append(pred[0])
        actuals.append(y_test[0])
        names.append(df.iloc[test_idx[0]]['Name'])
        groups.append(df.iloc[test_idx[0]]['Group'])
    
    return pd.DataFrame({
        'Name': names,
        'Group': groups,
        'Actual': actuals,
        'Predicted': predictions,
        'Error': np.array(actuals) - np.array(predictions),
        'Abs_Error': np.abs(np.array(actuals) - np.array(predictions))
    })

def plot_individual_predictions(results_df):
    """Create a comprehensive visualization of individual predictions"""
    
    # Sort by actual score
    results_df = results_df.sort_values('Actual', ascending=True)
    
    # Color mapping for groups
    group_colors = {
        'neutral': '#FFB84D',
        'opposing': '#4ECDC4', 
        'similar': '#FF6B6B'
    }
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ============ Plot 1: Bar chart comparing actual vs predicted ============
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(results_df))
    width = 0.35
    
    colors_actual = [group_colors[g.lower()] for g in results_df['Group']]
    
    bars1 = ax1.bar(x - width/2, results_df['Actual'], width, 
                     label='Actual Score', alpha=0.8, color=colors_actual,
                     edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, results_df['Predicted'], width,
                     label='Predicted Score', alpha=0.6, color='gray',
                     edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Participant', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Summary Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Actual vs Predicted Summary Scores\n(Ridge Regression - Fusion Data)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Name'], rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=results_df['Actual'].mean(), color='red', linestyle='--', 
                linewidth=2, label='Mean Actual', alpha=0.7)
    
    # ============ Plot 2: Scatter plot with perfect prediction line ============
    ax2 = plt.subplot(2, 2, 2)
    
    for group in results_df['Group'].unique():
        group_data = results_df[results_df['Group'] == group]
        ax2.scatter(group_data['Actual'], group_data['Predicted'], 
                   s=150, alpha=0.7, label=group.capitalize(),
                   color=group_colors[group.lower()],
                   edgecolors='black', linewidth=1.5)
    
    # Perfect prediction line
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, 
             label='Perfect Prediction', alpha=0.5)
    
    # Calculate and display correlation
    corr, p_value = stats.pearsonr(results_df['Actual'], results_df['Predicted'])
    ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.4f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Actual Score (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Predicted Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Accuracy by Group', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ============ Plot 3: Prediction errors by person ============
    ax3 = plt.subplot(2, 2, 3)
    
    colors_error = [group_colors[g.lower()] for g in results_df['Group']]
    bars = ax3.barh(results_df['Name'], results_df['Error'], 
                     color=colors_error, alpha=0.7,
                     edgecolor='black', linewidth=1)
    
    # Color negative errors differently
    for i, (bar, error) in enumerate(zip(bars, results_df['Error'])):
        if error < 0:
            bar.set_color('darkred')
            bar.set_alpha(0.5)
    
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax3.set_xlabel('Prediction Error (Actual - Predicted) %', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Participant', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Errors by Participant\n(Red = Overpredicted, Colored = Underpredicted)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # ============ Plot 4: Error statistics by group ============
    ax4 = plt.subplot(2, 2, 4)
    
    group_stats = results_df.groupby('Group').agg({
        'Abs_Error': ['mean', 'std'],
        'Error': 'mean'
    }).round(2)
    
    groups = group_stats.index
    abs_errors = group_stats['Abs_Error']['mean']
    error_bias = group_stats['Error']['mean']
    error_std = group_stats['Abs_Error']['std']
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, abs_errors, width, label='Mean Absolute Error',
                    color=[group_colors[g.lower()] for g in groups], alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, error_std, width, label='Std Dev of Error',
                    color='gray', alpha=0.6, edgecolor='black', linewidth=1.5)
    
    ax4.set_xlabel('Group', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Prediction Error Statistics by Group', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([g.capitalize() for g in groups])
    ax4.legend(fontsize=11)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('individual_predictions_detailed.png', dpi=300, bbox_inches='tight')
    print("✅ Detailed predictions plot saved to: individual_predictions_detailed.png")
    plt.close()

def plot_simple_comparison(results_df):
    """Create a simple, clean comparison chart"""
    
    # Sort by group and then by actual score
    results_df = results_df.sort_values(['Group', 'Actual'])
    
    # Color mapping for groups
    group_colors = {
        'neutral': '#FFB84D',
        'opposing': '#4ECDC4', 
        'similar': '#FF6B6B'
    }
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(results_df))
    width = 0.4
    
    colors_actual = [group_colors[g.lower()] for g in results_df['Group']]
    
    # Plot bars
    bars1 = ax.bar(x - width/2, results_df['Actual'], width, 
                   label='Actual Score', color=colors_actual, alpha=0.85,
                   edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, results_df['Predicted'], width,
                   label='Predicted Score', color='lightgray', alpha=0.85,
                   edgecolor='black', linewidth=1.5, hatch='//')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=8, 
               fontweight='bold', color='darkred')
    
    # Formatting
    ax.set_xlabel('Participant', fontsize=14, fontweight='bold')
    ax.set_ylabel('Summary Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Individual Predictions: Actual vs Predicted Summary Scores\nBest Model: Ridge Regression (LOOCV) | Fusion Data | R² = 0.563, r = 0.770***', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Name'], rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add group dividers
    group_changes = results_df['Group'].ne(results_df['Group'].shift()).cumsum()
    for i in range(1, group_changes.max()):
        idx = results_df[group_changes == i].index[0]
        pos = list(results_df.index).index(idx)
        ax.axvline(x=pos - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)
    
    # Add group labels
    for group in ['neutral', 'opposing', 'similar']:
        group_data = results_df[results_df['Group'] == group]
        if len(group_data) > 0:
            start_idx = list(results_df.index).index(group_data.index[0])
            end_idx = list(results_df.index).index(group_data.index[-1])
            mid_idx = (start_idx + end_idx) / 2
            ax.text(mid_idx, ax.get_ylim()[1] * 0.95, group.upper(), 
                   fontsize=12, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor=group_colors[group], alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('individual_predictions_simple.png', dpi=300, bbox_inches='tight')
    print("✅ Simple predictions chart saved to: individual_predictions_simple.png")
    plt.close()

def create_prediction_table(results_df):
    """Create a nicely formatted table"""
    results_df = results_df.sort_values(['Group', 'Actual'], ascending=[True, False])
    
    # Add accuracy indicator
    results_df['Accuracy'] = results_df.apply(
        lambda row: '✓ Excellent' if row['Abs_Error'] < 5 
        else '✓ Good' if row['Abs_Error'] < 10
        else '○ Fair' if row['Abs_Error'] < 15
        else '✗ Poor', axis=1
    )
    
    print("\n" + "="*100)
    print("INDIVIDUAL PREDICTION RESULTS - Ridge Regression (Fusion Data)")
    print("="*100)
    print(f"{'Name':<18} {'Group':<10} {'Actual':>8} {'Predicted':>10} {'Error':>8} {'Abs Error':>10} {'Accuracy':<12}")
    print("-"*100)
    
    for _, row in results_df.iterrows():
        print(f"{row['Name']:<18} {row['Group'].capitalize():<10} {row['Actual']:>8.1f} "
              f"{row['Predicted']:>10.1f} {row['Error']:>8.1f} {row['Abs_Error']:>10.1f} {row['Accuracy']:<12}")
    
    print("-"*100)
    print(f"\nOverall Statistics:")
    print(f"  Mean Absolute Error:  {results_df['Abs_Error'].mean():.2f}%")
    print(f"  RMSE:                 {np.sqrt((results_df['Error']**2).mean()):.2f}%")
    print(f"  Correlation (r):      {stats.pearsonr(results_df['Actual'], results_df['Predicted'])[0]:.3f}")
    print("="*100 + "\n")
    
    # Save to CSV
    results_df.to_csv('individual_predictions.csv', index=False)
    print("✅ Predictions table saved to: individual_predictions.csv\n")

def main():
    print("\n" + "="*80)
    print("INDIVIDUAL PREDICTIONS VISUALIZATION")
    print("Best Model: Ridge Regression with Fusion Data")
    print("="*80)
    
    # Load data
    df = load_data()
    features = get_top_features()
    
    print(f"\n📊 Running Leave-One-Out Cross-Validation...")
    print(f"   - {len(df)} participants")
    print(f"   - {len(features)} features")
    print(f"   - Model: Ridge Regression (alpha=1.0)")
    
    # Get predictions
    results_df = run_loocv_prediction(df, features)
    
    # Create visualizations
    print(f"\n🎨 Creating visualizations...")
    plot_simple_comparison(results_df)
    plot_individual_predictions(results_df)
    create_prediction_table(results_df)
    
    print("="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 individual_predictions_simple.png - Clean comparison chart")
    print("  📊 individual_predictions_detailed.png - Comprehensive 4-panel analysis")
    print("  📄 individual_predictions.csv - Detailed predictions table")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
