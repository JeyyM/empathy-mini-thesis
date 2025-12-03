#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Validation Prediction Model for Summary Scores
Uses Leave-One-Out Cross-Validation (LOOCV) to predict summary scores
based on emotional features from facial, voice, and fusion data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def load_data(modality='fusion'):
    """Load the merged data for specified modality"""
    data_path = f'../data/correlation_data/{modality}_summary_merged.csv'
    df = pd.read_csv(data_path)
    return df

def select_top_features(df, target_col='Overall_Percentage', n_features=10):
    """Select top N features based on Spearman correlation with target"""
    # Exclude target variables and metadata columns
    exclude_cols = [
        'Name', 'Group', 'name', 'group',
        'Overall_Percentage', 'Semantic_Similarity', 'Topic_Coverage', 'Factual_Accuracy',
        'Letter_Grade', 'Summary_Words', 'Original_Words', 'Compression_Ratio'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Only keep numeric columns
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    correlations = []
    for col in feature_cols:
        # Remove NaN values
        mask = ~(pd.isna(df[col]) | pd.isna(df[target_col]))
        if mask.sum() < 3:
            continue
        
        try:
            rho, p = stats.spearmanr(df[col][mask], df[target_col][mask])
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
    
    # Select top N features
    top_features = corr_df.head(n_features)['feature'].tolist()
    
    return top_features, corr_df

def prepare_data(df, features, target_col='Overall_Percentage'):
    """Prepare feature matrix and target vector"""
    X = df[features].values
    y = df[target_col].values
    
    # Handle missing values by imputing with mean
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    return X, y, imputer

def evaluate_models_loocv(X, y, feature_names):
    """Evaluate multiple models using Leave-One-Out Cross-Validation"""
    print("\n" + "="*80)
    print("LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
    print("="*80)
    
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, epsilon=0.1)
    }
    
    loo = LeaveOneOut()
    scaler = StandardScaler()
    
    results = []
    
    for name, model in models.items():
        print(f"\n{'─'*80}")
        print(f"Model: {name}")
        print(f"{'─'*80}")
        
        # Cross-validation predictions
        y_pred = []
        y_true = []
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and predict
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            
            y_pred.append(pred[0])
            y_true.append(y_test[0])
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Correlation between predicted and actual
        corr, p_value = stats.pearsonr(y_true, y_pred)
        
        print(f"  RMSE: {rmse:.3f}%")
        print(f"  MAE:  {mae:.3f}%")
        print(f"  R²:   {r2:.3f}")
        print(f"  Correlation: r = {corr:.3f} (p = {p_value:.4f})")
        
        results.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Correlation': corr,
            'P_Value': p_value,
            'Predictions': y_pred.copy(),
            'Actuals': y_true.copy()
        })
    
    return results

def plot_predictions(results, output_file='prediction_results.png'):
    """Plot actual vs predicted for all models"""
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        y_true = result['Actuals']
        y_pred = result['Predictions']
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=100, edgecolors='black', linewidth=1)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_true, p(y_true), 'b-', linewidth=2, alpha=0.5, label='Regression Line')
        
        # Labels and title
        ax.set_xlabel('Actual Score (%)', fontsize=10)
        ax.set_ylabel('Predicted Score (%)', fontsize=10)
        ax.set_title(f"{result['Model']}\nR² = {result['R2']:.3f}, RMSE = {result['RMSE']:.2f}%", 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Prediction plots saved to: {output_file}")
    plt.close()

def plot_feature_importance(model_name, features, X, y):
    """Plot feature importance for tree-based models"""
    if 'Random Forest' in model_name or 'Gradient Boosting' in model_name:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if 'Random Forest' in model_name:
            model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=42)
        
        model.fit(X_scaled, y)
        importances = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Feature importance plot saved to: {filename}")
        plt.close()

def create_comparison_table(results, output_file='model_comparison.csv'):
    """Create a comparison table of all models"""
    comparison_df = pd.DataFrame([
        {
            'Model': r['Model'],
            'RMSE': r['RMSE'],
            'MAE': r['MAE'],
            'R²': r['R2'],
            'Correlation': r['Correlation'],
            'P_Value': r['P_Value']
        }
        for r in results
    ])
    
    # Sort by R²
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    comparison_df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print(f"\n✅ Comparison table saved to: {output_file}")
    
    return comparison_df

def analyze_prediction_errors(results, df, feature_names):
    """Analyze prediction errors by group"""
    best_model = max(results, key=lambda x: x['R2'])
    
    print("\n" + "="*80)
    print(f"PREDICTION ERROR ANALYSIS - Best Model: {best_model['Model']}")
    print("="*80)
    
    # Create error analysis DataFrame
    error_df = pd.DataFrame({
        'Name': df['Name'].values,
        'Group': df['Group'].values,
        'Actual': best_model['Actuals'],
        'Predicted': best_model['Predictions'],
        'Error': best_model['Actuals'] - best_model['Predictions'],
        'Abs_Error': np.abs(best_model['Actuals'] - best_model['Predictions'])
    })
    
    print("\nIndividual Predictions:")
    print(error_df.to_string(index=False))
    
    print("\nError Statistics by Group:")
    group_stats = error_df.groupby('Group').agg({
        'Abs_Error': ['mean', 'std', 'min', 'max'],
        'Error': ['mean']
    }).round(3)
    print(group_stats)
    
    # Save error analysis
    error_df.to_csv('prediction_errors.csv', index=False)
    print(f"\n✅ Error analysis saved to: prediction_errors.csv")
    
    return error_df

def run_multimodal_comparison():
    """Compare prediction performance across all modalities"""
    print("\n" + "="*80)
    print("MULTIMODAL PREDICTION COMPARISON")
    print("="*80)
    
    modalities = ['facial', 'voice', 'fusion']
    multimodal_results = []
    
    for modality in modalities:
        print(f"\n{'='*80}")
        print(f"Analyzing {modality.upper()} modality...")
        print(f"{'='*80}")
        
        # Load data
        df = load_data(modality)
        
        # Select top features
        top_features, corr_df = select_top_features(df, n_features=10)
        
        print(f"\nTop 10 features selected based on correlation:")
        for idx, feat in enumerate(top_features, 1):
            feat_info = corr_df[corr_df['feature'] == feat].iloc[0]
            print(f"  {idx}. {feat}: ρ = {feat_info['rho']:.3f} (p = {feat_info['p_value']:.4f})")
        
        # Prepare data
        X, y, imputer = prepare_data(df, top_features)
        
        # Evaluate models
        results = evaluate_models_loocv(X, y, top_features)
        
        # Get best model
        best_model = max(results, key=lambda x: x['R2'])
        
        multimodal_results.append({
            'Modality': modality.capitalize(),
            'Best_Model': best_model['Model'],
            'RMSE': best_model['RMSE'],
            'MAE': best_model['MAE'],
            'R2': best_model['R2'],
            'Correlation': best_model['Correlation']
        })
    
    # Create comparison
    multimodal_df = pd.DataFrame(multimodal_results)
    multimodal_df.to_csv('multimodal_comparison.csv', index=False)
    
    print("\n" + "="*80)
    print("MULTIMODAL PERFORMANCE COMPARISON")
    print("="*80)
    print(multimodal_df.to_string(index=False))
    print(f"\n✅ Multimodal comparison saved to: multimodal_comparison.csv")
    
    return multimodal_df

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("SUMMARY SCORE PREDICTION MODEL")
    print("Cross-Validation Analysis")
    print("="*80)
    print("Method: Leave-One-Out Cross-Validation (LOOCV)")
    print("Sample Size: 15 participants")
    print("Target: Overall Summary Score (%)")
    print("="*80)
    
    # Ask user for modality
    print("\nAvailable modalities:")
    print("  1. Facial")
    print("  2. Voice")
    print("  3. Fusion")
    print("  4. Compare all modalities")
    
    # For now, default to fusion (you can modify this)
    modality = 'fusion'
    print(f"\nUsing {modality.upper()} modality for detailed analysis...")
    
    # Load data
    df = load_data(modality)
    print(f"\n✅ Loaded {len(df)} participants with {len(df.columns)} features")
    
    # Select top features
    n_features = 15  # Use top 15 features
    top_features, corr_df = select_top_features(df, n_features=n_features)
    
    print(f"\n📊 Top {n_features} features selected based on Spearman correlation:")
    for idx, feat in enumerate(top_features, 1):
        feat_info = corr_df[corr_df['feature'] == feat].iloc[0]
        sig = '***' if feat_info['p_value'] < 0.001 else '**' if feat_info['p_value'] < 0.01 else '*' if feat_info['p_value'] < 0.05 else ''
        print(f"  {idx:2d}. {feat:40s} | ρ = {feat_info['rho']:7.3f} | p = {feat_info['p_value']:.4f} {sig}")
    
    # Prepare data
    X, y, imputer = prepare_data(df, top_features)
    
    # Evaluate models
    results = evaluate_models_loocv(X, y, top_features)
    
    # Plot predictions
    plot_predictions(results, 'prediction_results_fusion.png')
    
    # Create comparison table
    comparison_df = create_comparison_table(results, 'model_comparison_fusion.csv')
    
    # Plot feature importance for best tree-based model
    tree_models = [r for r in results if 'Random Forest' in r['Model'] or 'Gradient Boosting' in r['Model']]
    if tree_models:
        best_tree = max(tree_models, key=lambda x: x['R2'])
        plot_feature_importance(best_tree['Model'], top_features, X, y)
    
    # Analyze prediction errors
    error_df = analyze_prediction_errors(results, df, top_features)
    
    # Run multimodal comparison
    print("\n" + "="*80)
    print("Running multimodal comparison...")
    print("="*80)
    multimodal_df = run_multimodal_comparison()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 prediction_results_fusion.png - Actual vs Predicted plots")
    print("  📊 feature_importance_*.png - Feature importance plots")
    print("  📄 model_comparison_fusion.csv - Model performance comparison")
    print("  📄 prediction_errors.csv - Individual prediction errors")
    print("  📄 multimodal_comparison.csv - Comparison across modalities")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
