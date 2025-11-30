"""
Quick Analysis Runner - Personality-Empathy Correlations
Run this anytime you update the data or want to re-run the analysis
"""

import subprocess
import sys

print("=" * 80)
print("RUNNING PERSONALITY-EMPATHY CORRELATION ANALYSIS")
print("=" * 80)
print()
print("This script will:")
print("  1. Match Google Form responses with emotion analysis results")
print("  2. Calculate correlations between personality traits and emotions")
print("  3. Generate 8 output files with visualizations and reports")
print()
print("Expected outputs in 'results/correlation_analysis/':")
print("  - master_dataset.csv (combined data)")
print("  - all_correlations.csv (all 194 correlations)")
print("  - 5 visualization PNG files")
print("  - correlation_report.txt (detailed technical report)")
print()
input("Press Enter to start analysis...")
print()

# Run the main analysis script
result = subprocess.run([sys.executable, "analyze_personality_empathy_correlation.py"])

if result.returncode == 0:
    print()
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("📖 Read the summary: CORRELATION_ANALYSIS_SUMMARY.md")
    print("📊 View visualizations: results/correlation_analysis/*.png")
    print("📁 View raw data: results/correlation_analysis/*.csv")
else:
    print()
    print("❌ Analysis failed. Check error messages above.")
    sys.exit(1)
