#!/usr/bin/env python3
"""
Enhanced prediction model integrating personality traits with emotional features
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# UTF-8 encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("PERSONALITY-ENHANCED PREDICTION MODEL")
print("="*80)

# Load emotion data
emotion_df = pd.read_csv('../correlation/fusion_summary_merged.csv')
print(f"\nEmotion data loaded: {emotion_df.shape[0]} participants, {emotion_df.shape[1]} features")

# Load personality data
personality_df = pd.read_csv('../personality/forms_responses/Multimodal Analysis of Empathy in Opposing View Dialogues (Responses) - Form Responses 1.csv')
print(f"Personality data loaded: {personality_df.shape[0]} responses")

# Extract personality traits
personality_traits = [
    'I express my thoughts clearly during conversations.',
    'I feel comfortable talking about controversial or sensitive topics.',
    'I listen more than I talk during most conversations.',
    'I am comfortable expressing emotions in conversations.',
    'I have strong opinions that I rarely change.',
    'I tend to stay calm even when conversations become hard or frustrating.',
    'I tend to make the other person feel heard during conversations.',
    'I handle disagreements well.'
]

# Process personality data
personality_processed = personality_df[['Full name'] + personality_traits].copy()
personality_processed.columns = ['Name'] + [f'personality_{i+1}' for i in range(len(personality_traits))]

# Clean names for matching
def clean_name(name):
    """Clean name for matching"""
    if pd.isna(name):
        return ""
    # Remove extra spaces, convert to title case
    name = ' '.join(name.split())
    # Handle special cases
    name_mapping = {
        'Miguel Zakia Ng': 'MiguelNg',
        'Ryan Justin So': 'RyanSo',
        'Sean Te': 'SeanTe',
        'Anton Miguel G. Borromeo': 'MiguelBorromeo',
        'Louise Randell-so R. Fabico': 'RandellFabico',
        'Russell Emmanuel G. Galan': 'RusselGalan',
        'Aaron Dionisio': 'AaronDionisio',
        'Charlz Ivhan S. Pates': 'ArianPates',
        'Samuel Lim': 'SamuelLim',
        'Ethan Brook Ong': 'EthanOng',
        'Keithzi Rhaz Cantona': 'KeithziCantona',
        'Maggie Chong': 'MaggieOng',
        'Marwah B. Muti': 'MarwahMuti',
        'Joaquin Adrien U. Ong': 'AndreMarco',
        'Ethan Plaza': 'EthanPlaza'
    }
    
    for full, short in name_mapping.items():
        if full.lower() == name.lower():
            return short
    
    return name

personality_processed['Name'] = personality_processed['Name'].apply(clean_name)

# Merge with emotion data
merged_df = emotion_df.merge(personality_processed, on='Name', how='left')
print(f"\nMerged data: {merged_df.shape[0]} participants, {merged_df.shape[1]} total features")
print(f"Personality traits matched: {merged_df[['personality_1']].notna().sum()[0]}/{len(merged_df)}")

# Check for missing personality data
missing_personality = merged_df[merged_df['personality_1'].isna()]['Name'].tolist()
if missing_personality:
    print(f"\n⚠ WARNING: Missing personality data for: {', '.join(missing_personality)}")

# Target variable
target = 'Overall_Percentage'
y = merged_df[target].values

# Prepare feature sets
excluded_cols = ['Group', 'Name', 'Overall_Percentage', 'Letter_Grade', 
                 'Semantic_Similarity', 'Topic_Coverage', 'Factual_Accuracy',
                 'Original_Words', 'Summary_Words', 'Compression_Ratio',
                 'fused_dominant_emotion']

# 1. Emotional features only (top 15 from previous analysis)
# Using features that exist in the fusion dataset
top_emotional_features = [
    'combined_quadrant_transitions',
    'fused_quadrant_transitions',
    'disgust_total_change',
    'combined_valence_total_change',
    'disgust_max_change',
    'combined_arousal_total_change',
    'disgust_mean_change',
    'arousal_total_change',
    'valence_total_change',
    'disgust_trend_slope',
    'combined_arousal_mean_change',
    'arousal_mean_change',
    'combined_valence_mean_change',
    'combined_intensity_total_change',
    'fused_overall_volatility'
]

# 2. Personality features
personality_features = [f'personality_{i+1}' for i in range(8)]

# 3. Combined features
combined_features = top_emotional_features + personality_features

print("\n" + "="*80)
print("FEATURE SETS:")
print("="*80)
print(f"Emotional features: {len(top_emotional_features)}")
print(f"Personality features: {len(personality_features)}")
print(f"Combined features: {len(combined_features)}")

# Function to train and evaluate models
def evaluate_models(X, y, feature_names, dataset_name):
    """Evaluate multiple models using LOOCV"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {dataset_name}")
    print(f"{'='*80}")
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Models to evaluate
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    results = {}
    
    for model_name, model in models.items():
        loo = LeaveOneOut()
        predictions = []
        actuals = []
        
        for train_idx, test_idx in loo.split(X_imputed):
            X_train, X_test = X_imputed[train_idx], X_imputed[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and predict
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)[0]
            
            predictions.append(pred)
            actuals.append(y_test[0])
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        r, p_value = stats.pearsonr(actuals, predictions)
        r2 = r**2
        
        results[model_name] = {
            'predictions': predictions,
            'actuals': actuals,
            'MAE': mae,
            'RMSE': rmse,
            'r': r,
            'p_value': p_value,
            'R2': r2
        }
        
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        print(f"\n{model_name}:")
        print(f"  MAE:  {mae:.2f}%")
        print(f"  RMSE: {rmse:.2f}%")
        print(f"  r:    {r:.3f}{sig} (p={p_value:.6f})")
        print(f"  R²:   {r2:.3f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['R2'])
    print(f"\n{'='*40}")
    print(f"BEST MODEL: {best_model[0]}")
    print(f"{'='*40}")
    
    return results, best_model[0]

# Evaluate all three feature sets
print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# 1. Emotional features only
X_emotional = merged_df[top_emotional_features].values
results_emotional, best_emotional = evaluate_models(
    X_emotional, y, top_emotional_features, 
    "EMOTIONAL FEATURES ONLY (baseline)"
)

# 2. Personality features only  
X_personality = merged_df[personality_features].values
results_personality, best_personality = evaluate_models(
    X_personality, y, personality_features,
    "PERSONALITY FEATURES ONLY"
)

# 3. Combined features
X_combined = merged_df[combined_features].values
results_combined, best_combined = evaluate_models(
    X_combined, y, combined_features,
    "COMBINED (Emotional + Personality)"
)

# Compare best models from each feature set
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

comparison_data = {
    'Feature Set': ['Emotional Only', 'Personality Only', 'Combined'],
    'Best Model': [best_emotional, best_personality, best_combined],
    'MAE': [
        results_emotional[best_emotional]['MAE'],
        results_personality[best_personality]['MAE'],
        results_combined[best_combined]['MAE']
    ],
    'RMSE': [
        results_emotional[best_emotional]['RMSE'],
        results_personality[best_personality]['RMSE'],
        results_combined[best_combined]['RMSE']
    ],
    'r': [
        results_emotional[best_emotional]['r'],
        results_personality[best_personality]['r'],
        results_combined[best_combined]['r']
    ],
    'R²': [
        results_emotional[best_emotional]['R2'],
        results_personality[best_personality]['R2'],
        results_combined[best_combined]['R2']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Calculate improvement
baseline_r2 = results_emotional[best_emotional]['R2']
combined_r2 = results_combined[best_combined]['R2']
improvement = ((combined_r2 - baseline_r2) / baseline_r2) * 100

print(f"\n{'='*80}")
print("KEY FINDINGS:")
print(f"{'='*80}")
print(f"Baseline (Emotional only) R²: {baseline_r2:.3f}")
print(f"Combined R²: {combined_r2:.3f}")
print(f"Improvement: {improvement:+.1f}%")

if combined_r2 > baseline_r2:
    print(f"\n✓ Personality features IMPROVED prediction accuracy!")
elif combined_r2 < baseline_r2:
    print(f"\n✗ Personality features DECREASED prediction accuracy")
else:
    print(f"\n- No change in prediction accuracy")

# Save results
output_df = pd.DataFrame({
    'Name': merged_df['Name'],
    'Group': merged_df['Group'],
    'Actual': y,
    'Predicted_Emotional': results_emotional[best_emotional]['predictions'],
    'Predicted_Personality': results_personality[best_personality]['predictions'],
    'Predicted_Combined': results_combined[best_combined]['predictions'],
    'Error_Emotional': results_emotional[best_emotional]['predictions'] - y,
    'Error_Personality': results_personality[best_personality]['predictions'] - y,
    'Error_Combined': results_combined[best_combined]['predictions'] - y
})

output_df.to_csv('personality_enhanced_predictions.csv', index=False)
print(f"\n✓ Results saved to: personality_enhanced_predictions.csv")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Personality-Enhanced Prediction Model Comparison', fontsize=16, fontweight='bold')

# Plot 1: Emotional only
ax1 = axes[0, 0]
ax1.scatter(results_emotional[best_emotional]['actuals'], 
           results_emotional[best_emotional]['predictions'], 
           c='blue', alpha=0.6, s=100, edgecolors='black')
ax1.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect prediction')
ax1.set_xlabel('Actual Score (%)', fontsize=11)
ax1.set_ylabel('Predicted Score (%)', fontsize=11)
ax1.set_title(f'Emotional Features Only\n{best_emotional}: R²={results_emotional[best_emotional]["R2"]:.3f}', 
             fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Personality only
ax2 = axes[0, 1]
ax2.scatter(results_personality[best_personality]['actuals'],
           results_personality[best_personality]['predictions'],
           c='green', alpha=0.6, s=100, edgecolors='black')
ax2.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect prediction')
ax2.set_xlabel('Actual Score (%)', fontsize=11)
ax2.set_ylabel('Predicted Score (%)', fontsize=11)
ax2.set_title(f'Personality Features Only\n{best_personality}: R²={results_personality[best_personality]["R2"]:.3f}',
             fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Combined
ax3 = axes[1, 0]
ax3.scatter(results_combined[best_combined]['actuals'],
           results_combined[best_combined]['predictions'],
           c='purple', alpha=0.6, s=100, edgecolors='black')
ax3.plot([0, 100], [0, 100], 'r--', lw=2, label='Perfect prediction')
ax3.set_xlabel('Actual Score (%)', fontsize=11)
ax3.set_ylabel('Predicted Score (%)', fontsize=11)
ax3.set_title(f'Combined Features\n{best_combined}: R²={results_combined[best_combined]["R2"]:.3f}',
             fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: R² comparison
ax4 = axes[1, 1]
r2_values = [baseline_r2, results_personality[best_personality]['R2'], combined_r2]
bars = ax4.bar(['Emotional\nOnly', 'Personality\nOnly', 'Combined'],
               r2_values,
               color=['blue', 'green', 'purple'],
               alpha=0.7,
               edgecolor='black')
ax4.set_ylabel('R² Score', fontsize=11)
ax4.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 1)
ax4.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('personality_enhanced_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: personality_enhanced_comparison.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
