#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Essential Group Comparison Analysis
Compares Neutral, Opposing, and Similar groups on:
- Facial emotions
- Voice emotions  
- Fusion emotions
- Emotional volatility
- Summary quality
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import run_essential_analysis, print_summary_report

if __name__ == "__main__":
    # Set path to workspace root (go up two levels from analysis folder)
    workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    print(f"Working directory: {workspace_path}")
    
    # Run analysis
    results = run_essential_analysis(workspace_path)
    
    # Print summary report
    print_summary_report(results)
    
    # Print category breakdown
    print("\n" + "="*80)
    print("ANALYSIS BREAKDOWN BY CATEGORY")
    print("="*80)
    
    for category in results['Category'].unique():
        cat_results = results[results['Category'] == category]
        sig_count = len(cat_results[cat_results['P_Value'] < 0.05])
        total_count = len(cat_results)
        
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  Total features tested: {total_count}")
        print(f"  Significant (p<0.05): {sig_count} ({sig_count/total_count*100:.1f}%)")
        
        if sig_count > 0:
            print(f"  Significant features:")
            sig_features = cat_results[cat_results['P_Value'] < 0.05]['Feature'].tolist()
            for feat in sig_features:
                sig_marker = cat_results[cat_results['Feature'] == feat]['Significant'].iloc[0]
                p_val = cat_results[cat_results['Feature'] == feat]['P_Value'].iloc[0]
                print(f"    - {feat} ({sig_marker}, p={p_val:.4f})")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nFull results saved to: {workspace_path}/analysis/group_comparison_results.csv")
    print("\nLegend:")
    print("  *** p < 0.001 (highly significant)")
    print("  **  p < 0.01  (very significant)")
    print("  *   p < 0.05  (significant)")
    print("  ns  p >= 0.05 (not significant)")
    print("="*80 + "\n")
