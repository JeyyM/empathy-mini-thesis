#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a clean table visualization of group comparison results
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import os

# Find the CSV file
if os.path.exists('group_comparison_results.csv'):
    csv_path = 'group_comparison_results.csv'
elif os.path.exists('../../group_comparison_results.csv'):
    csv_path = '../../group_comparison_results.csv'
elif os.path.exists('../../analysis/group_comparison_results.csv'):
    csv_path = '../../analysis/group_comparison_results.csv'
else:
    raise FileNotFoundError("Cannot find group_comparison_results.csv")

# Load results
print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# Sort by category and feature
df = df.sort_values(['Category', 'Feature'])

# Prepare data for table
table_data = []
for _, row in df.iterrows():
    neutral = row.get('neutral_mean', 0)
    opposing = row.get('opposing_mean', 0)
    similar = row.get('similar_mean', 0)
    
    # Format values based on category
    if 'Summary' in row['Category']:
        neutral_str = f"{neutral:.2f}"
        opposing_str = f"{opposing:.2f}"
        similar_str = f"{similar:.2f}"
    else:
        neutral_str = f"{neutral:.3f}"
        opposing_str = f"{opposing:.3f}"
        similar_str = f"{similar:.3f}"
    
    # Significance
    sig = row['Significant']
    if sig != 'ns':
        sig_str = f"✓ {sig}"
    else:
        sig_str = ""
    
    # Feature name with category
    feature_name = f"{row['Category'].replace('_', ' ')}: {row['Feature']}"
    
    table_data.append([feature_name, neutral_str, opposing_str, similar_str, sig_str])

# Create figure
fig = plt.figure(figsize=(16, 20))
ax = fig.add_subplot(111)
ax.axis('tight')
ax.axis('off')

# Column headers
col_headers = ['Feature', 'Neutral\nGroup', 'Opposing\nGroup', 'Similar\nGroup', 'Significant?']

# Create table
table = ax.table(cellText=table_data,
                 colLabels=col_headers,
                 cellLoc='left',
                 loc='center',
                 colWidths=[0.45, 0.12, 0.12, 0.12, 0.19])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Header styling
for i in range(len(col_headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', fontsize=11)
    cell.set_height(0.08)

# Cell styling
categories = []
current_category = None
row_idx = 1

for i, row_data in enumerate(table_data):
    # Alternate row colors by category
    category = row_data[0].split(':')[0]
    if category != current_category:
        current_category = category
        categories.append(category)
    
    # Determine row color based on category
    cat_idx = categories.index(category)
    if cat_idx % 2 == 0:
        row_color = '#F2F2F2'
    else:
        row_color = 'white'
    
    # Color the row
    for j in range(len(col_headers)):
        cell = table[(i+1, j)]
        cell.set_facecolor(row_color)
        cell.set_height(0.04)
        
        # Highlight significant results
        if j == 4 and row_data[4]:  # Significant column
            cell.set_facecolor('#C6E0B4')
            cell.set_text_props(weight='bold', color='#006100')

# Add title
plt.title('Group Comparison Results: Neutral vs Opposing vs Similar\n', 
          fontsize=16, fontweight='bold', pad=20)

# Add legend for significance
legend_text = (
    "Significance Levels:\n"
    "*** = p < 0.001 (Highly Significant)\n"
    "**  = p < 0.01  (Very Significant)\n"
    "*   = p < 0.05  (Significant)\n"
    "(blank) = p ≥ 0.05 (Not Significant)"
)

plt.text(0.02, 0.02, legend_text, transform=fig.transFigure, 
         fontsize=9, verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Save
plt.tight_layout()
plt.savefig('group_comparison_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Table saved to: group_comparison_table.png")
plt.close()

# Also create a simplified version with only trending results
print("\n" + "="*80)
print("Creating simplified table with trending results...")
print("="*80)

# Filter for p < 0.15
trending_df = df[df['P_Value'] < 0.15].sort_values('P_Value')

if len(trending_df) > 0:
    table_data_simple = []
    for _, row in trending_df.iterrows():
        neutral = row.get('neutral_mean', 0)
        opposing = row.get('opposing_mean', 0)
        similar = row.get('similar_mean', 0)
        
        # Format values
        if 'Summary' in row['Category']:
            neutral_str = f"{neutral:.2f}"
            opposing_str = f"{opposing:.2f}"
            similar_str = f"{similar:.2f}"
        else:
            neutral_str = f"{neutral:.3f}"
            opposing_str = f"{opposing:.3f}"
            similar_str = f"{similar:.3f}"
        
        # Feature name
        feature_name = f"{row['Category'].replace('_', ' ')}: {row['Feature']}"
        
        # Significance with p-value
        sig_str = f"p={row['P_Value']:.4f}"
        
        table_data_simple.append([feature_name, neutral_str, opposing_str, similar_str, sig_str])
    
    # Create simplified figure
    fig2 = plt.figure(figsize=(14, 6))
    ax2 = fig2.add_subplot(111)
    ax2.axis('tight')
    ax2.axis('off')
    
    # Create table
    table2 = ax2.table(cellText=table_data_simple,
                      colLabels=col_headers,
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.45, 0.12, 0.12, 0.12, 0.19])
    
    # Style
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 3)
    
    # Header styling
    for i in range(len(col_headers)):
        cell = table2[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Row styling with gradient based on p-value
    for i in range(len(table_data_simple)):
        p_val = trending_df.iloc[i]['P_Value']
        
        # Color based on significance
        if p_val < 0.05:
            row_color = '#C6E0B4'  # Green
        elif p_val < 0.10:
            row_color = '#FFE699'  # Yellow
        else:
            row_color = '#FFF2CC'  # Light yellow
        
        for j in range(len(col_headers)):
            cell = table2[(i+1, j)]
            cell.set_facecolor(row_color)
    
    plt.title('Trending Results: Group Differences (p < 0.15)\n', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    legend_text2 = (
        f"Found {len(trending_df)} trending results\n"
        "Green: p < 0.05 (Significant)\n"
        "Yellow: p < 0.10 (Trending)\n"
        "Light Yellow: p < 0.15 (Marginally Trending)"
    )
    
    plt.text(0.02, 0.02, legend_text2, transform=fig2.transFigure, 
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('group_comparison_table_trending.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Trending results table saved to: group_comparison_table_trending.png")
else:
    print("No trending results found (all p ≥ 0.15)")

print("\n" + "="*80)
print("✅ TABLE GENERATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  📊 group_comparison_table.png - Full comparison table")
print("  📊 group_comparison_table_trending.png - Trending results only")
print("="*80 + "\n")
