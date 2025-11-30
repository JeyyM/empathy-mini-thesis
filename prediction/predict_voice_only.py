#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice-Only Prediction Model for Summary Scores
Uses only voice emotion features to predict summary quality
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

print("="*80)
print("VOICE-ONLY PREDICTION MODEL")
print("="*80)
print("Using: Voice emotion features only")
print("Target: Overall Summary Score (%)")
print("Method: Leave-One-Out Cross-Validation (LOOCV)")
print("="*80)

# Load voice data
df = pd.read_csv('../correlation/voice_summary_merged.csv')
print(f"\n✅ Loaded {len(df)} participants with voice emotion data")

# Exclude target and metadata columns
exclude_cols = [
    'Name', 'Group', 'name', 'group',
    'Overall_Percentage', 'Semantic_Similarity', 'Topic_Coverage', 'Factual_Accuracy',
    'Letter_Grade', 'Summary_Words', 'Original_Words', 'Compression_Ratio'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]

print(f"Total voice features available: {len(feature_cols)}")

# Calculate correlations to select top features
correlations = []
for col in feature_cols:
    mask = ~(pd.isna(df[col]) | pd.isna(df['Overall_Percentage']))
    if mask.sum() < 3:
        continue
    
    try:
        rho, p = stats.spearmanr(df[col][mask], df['Overall_Percentage'][mask])
        correlations.append({
            'feature': col,
            'correlation': abs(rho),
            'rho': rho,
            'p_value': p
        })
    except:
        continue

corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('correlation', ascending=False)

# Select top 15 features
n_features = 15
top_features = corr_df.head(n_features)['feature'].tolist()

print(f"\n📊 Top {n_features} voice features selected:")
for idx, feat in enumerate(top_features, 1):
    feat_info = corr_df[corr_df['feature'] == feat].iloc[0]
    sig = '***' if feat_info['p_value'] < 0.001 else '**' if feat_info['p_value'] < 0.01 else '*' if feat_info['p_value'] < 0.05 else ''
    print(f"  {idx:2d}. {feat:40s} | ρ = {feat_info['rho']:7.3f} | p = {feat_info['p_value']:.4f} {sig}")

# Prepare data
X = df[top_features].values
y = df['Overall_Percentage'].values

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print("\n" + "="*80)
print("EVALUATING MODELS WITH VOICE FEATURES ONLY")
print("="*80)

models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

loo = LeaveOneOut()
scaler = StandardScaler()
results = []

for name, model in models.items():
    y_pred = []
    y_true = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        y_pred.append(pred[0])
        y_true.append(y_test[0])
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, p_value = stats.pearsonr(y_true, y_pred)
    
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f}%")
    print(f"  RMSE: {rmse:.2f}%")
    print(f"  r:    {corr:.3f}{sig} (p={p_value:.6f})")
    print(f"  R²:   {r2:.3f}")
    
    results.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'r': corr,
        'p_value': p_value,
        'R2': r2,
        'predictions': y_pred,
        'actuals': y_true
    })

# Find best model
best_model = max(results, key=lambda x: x['R2'])

print("\n" + "="*80)
print(f"BEST MODEL: {best_model['Model']}")
print("="*80)
print(f"MAE:  {best_model['MAE']:.2f}%")
print(f"RMSE: {best_model['RMSE']:.2f}%")
print(f"r:    {best_model['r']:.3f} (p={best_model['p_value']:.6f})")
print(f"R²:   {best_model['R2']:.3f}")

# Save predictions
pred_df = pd.DataFrame({
    'Name': df['Name'].values,
    'Group': df['Group'].values,
    'Actual': best_model['actuals'],
    'Predicted': best_model['predictions'],
    'Error': best_model['actuals'] - best_model['predictions'],
    'Abs_Error': np.abs(best_model['actuals'] - best_model['predictions'])
})

pred_df.to_csv('voice_only_predictions.csv', index=False)
print(f"\n✅ Predictions saved to: voice_only_predictions.csv")

# Create simple bar chart visualization (matching individual_predictions_simple style)
# Sort by group and then by actual score
results_sorted = pred_df.sort_values(['Group', 'Actual'])

# Color mapping for groups
group_colors = {
    'neutral': '#FFB84D',
    'opposing': '#4ECDC4', 
    'similar': '#FF6B6B'
}

fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(results_sorted))
width = 0.4

colors_actual = [group_colors[g.lower()] for g in results_sorted['Group']]

# Plot bars
bars1 = ax.bar(x - width/2, results_sorted['Actual'], width, 
               label='Actual Score', color=colors_actual, alpha=0.85,
               edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, results_sorted['Predicted'], width,
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
sig_marker = '***' if best_model['p_value'] < 0.001 else '**' if best_model['p_value'] < 0.01 else '*' if best_model['p_value'] < 0.05 else 'ns'
ax.set_title(f'Voice-Only Predictions: Actual vs Predicted Summary Scores\nBest Model: {best_model["Model"]} (LOOCV) | Voice Data | R² = {best_model["R2"]:.3f}, r = {best_model["r"]:.3f}{sig_marker}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(results_sorted['Name'], rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add group dividers
group_changes = results_sorted['Group'].ne(results_sorted['Group'].shift()).cumsum()
for i in range(1, group_changes.max()):
    idx = results_sorted[group_changes == i].index[0]
    pos = list(results_sorted.index).index(idx)
    ax.axvline(x=pos - 0.5, color='black', linestyle='--', linewidth=2, alpha=0.5)

# Add group labels
for group in ['neutral', 'opposing', 'similar']:
    group_data = results_sorted[results_sorted['Group'] == group]
    if len(group_data) > 0:
        start_idx = list(results_sorted.index).index(group_data.index[0])
        end_idx = list(results_sorted.index).index(group_data.index[-1])
        mid_idx = (start_idx + end_idx) / 2
        ax.text(mid_idx, ax.get_ylim()[1] * 0.95, group.upper(), 
               fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor=group_colors[group], alpha=0.3))

# Add group means in top right
group_stats = []
for group in ['Neutral', 'Opposing', 'Similar']:
    group_data = pred_df[pred_df['Group'] == group]
    if len(group_data) > 0:
        mean_actual = group_data['Actual'].mean()
        mean_predicted = group_data['Predicted'].mean()
        group_stats.append(f"{group}: Actual={mean_actual:.1f}%, Pred={mean_predicted:.1f}%")

stats_text = '\n'.join(group_stats)
ax.text(0.98, 0.85, stats_text, transform=ax.transAxes, 
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('voice_only_predictions.png', dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: voice_only_predictions.png")

print("\n" + "="*80)
print("✅ VOICE-ONLY MODEL COMPLETE!")
print("="*80)
