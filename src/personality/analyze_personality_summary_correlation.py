"""
Personality-Text Summary Correlation Analysis
Connects participants' self-reported communication personality traits with their text summarization quality scores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class PersonalitySummaryAnalyzer:
    def __init__(self, forms_csv_path, grading_csv_path):
        """
        Initialize analyzer with paths to data
        
        Args:
            forms_csv_path: Path to Google Forms responses CSV
            grading_csv_path: Path to text summarization grading_results.csv
        """
        self.forms_csv_path = forms_csv_path
        self.grading_csv_path = grading_csv_path
        
        # Define communication personality question columns
        self.personality_questions = {
            'clarity': 'I express my thoughts clearly during conversations.',
            'comfort_sensitive': 'I feel comfortable talking about controversial or sensitive topics.',
            'listening': 'I listen more than I talk during most conversations.',
            'express_emotions': 'I am comfortable expressing emotions in conversations.',
            'strong_opinions': 'I have strong opinions that I rarely change.',
            'stays_calm': 'I tend to stay calm even when conversations become hard or frustrating.',
            'makes_heard': 'I tend to make the other person feel heard during conversations.',
            'handles_disagreements': 'I handle disagreements well.'
        }
        
        # Name mapping (folder name to full name)
        self.name_map = {
            'MiguelBorromeo': 'Anton Miguel G. Borromeo',
            'MiguelNg': 'Miguel Zakia Ng',
            'RandellFabico': 'Louise Randell-so R. Fabico',
            'RusselGalan': 'Russell Emmanuel G. Galan',
            'RyanSo': 'Ryan Justin So',
            'AaronDionisio': 'Aaron Dionisio',
            'ArianPates': 'Charlz Arian S. Pates',
            'EthanPlaza': 'Ethan Luise Benedict C. Plaza',
            'MarwahMuti': 'Marwah B. Muti',
            'SamuelLim': 'Samuel Lim',
            'AndreMarco': 'Pete Andre Marco',
            'EthanOng': 'Ethan Brook Ong',
            'KeithziCantona': 'Keithzi Rhaz Cantona',
            'MaggieOng': 'Maggie Chong',
            'SeanTe': 'Sean Te'
        }
        
    def load_forms_data(self):
        """Load and process Google Forms responses"""
        print("Loading Google Forms responses...")
        df = pd.read_csv(self.forms_csv_path)
        
        # Extract relevant columns
        personality_cols = list(self.personality_questions.values())
        self.forms_df = df[['Email Address (registration and contact)', 'Full name'] + personality_cols].copy()
        
        # Rename columns for easier access
        col_rename = {'Email Address (registration and contact)': 'email', 'Full name': 'full_name'}
        for short_name, long_name in self.personality_questions.items():
            col_rename[long_name] = short_name
        
        self.forms_df.rename(columns=col_rename, inplace=True)
        
        # Clean full names
        self.forms_df['full_name'] = self.forms_df['full_name'].str.strip()
        
        print(f"✓ Loaded {len(self.forms_df)} form responses")
        return self.forms_df
    
    def load_grading_data(self):
        """Load text summarization grading results"""
        print("\nLoading grading results...")
        self.grading_df = pd.read_csv(self.grading_csv_path)
        
        # Map folder names to full names
        self.grading_df['full_name'] = self.grading_df['Name'].map(self.name_map)
        
        print(f"✓ Loaded {len(self.grading_df)} grading results")
        print(f"  Columns: {', '.join(self.grading_df.columns.tolist())}")
        return self.grading_df
    
    def create_master_dataset(self):
        """Merge personality traits with text summary scores"""
        print("\nMerging personality and grading data...")
        
        # Merge on full_name
        self.master_df = pd.merge(
            self.forms_df,
            self.grading_df,
            on='full_name',
            how='inner'
        )
        
        print(f"✓ Master dataset created with {len(self.master_df)} participants")
        print(f"  Matched participants: {', '.join(self.master_df['full_name'].tolist())}")
        
        # Show any missing participants
        missing_in_grading = set(self.forms_df['full_name']) - set(self.master_df['full_name'])
        missing_in_forms = set(self.grading_df['full_name'].dropna()) - set(self.master_df['full_name'])
        
        if missing_in_grading:
            print(f"\n  ⚠ Missing in grading data: {', '.join(missing_in_grading)}")
        if missing_in_forms:
            print(f"  ⚠ Missing in forms data: {', '.join(missing_in_forms)}")
        
        return self.master_df
    
    def calculate_correlations(self):
        """Calculate correlations between personality traits and summary scores"""
        print("\nCalculating correlations...")
        
        personality_cols = list(self.personality_questions.keys())
        summary_cols = ['Overall_Percentage', 'Semantic_Similarity', 'Topic_Coverage', 
                       'Factual_Accuracy', 'Compression_Ratio']
        
        # Calculate Pearson and Spearman correlations
        correlations = {}
        
        for personality in personality_cols:
            correlations[personality] = {}
            for summary_metric in summary_cols:
                # Pearson correlation
                pearson_r, pearson_p = stats.pearsonr(
                    self.master_df[personality], 
                    self.master_df[summary_metric]
                )
                
                # Spearman correlation (for ordinal data)
                spearman_r, spearman_p = stats.spearmanr(
                    self.master_df[personality], 
                    self.master_df[summary_metric]
                )
                
                correlations[personality][summary_metric] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
        
        self.correlations = correlations
        print("✓ Correlations calculated")
        return correlations
    
    def find_significant_correlations(self, alpha=0.05):
        """Find statistically significant correlations"""
        print(f"\nFinding significant correlations (α = {alpha})...")
        
        significant = []
        
        for personality, summary_dict in self.correlations.items():
            for summary_metric, stats_dict in summary_dict.items():
                if stats_dict['pearson_p'] < alpha:
                    significant.append({
                        'personality_trait': personality,
                        'summary_metric': summary_metric,
                        'pearson_r': stats_dict['pearson_r'],
                        'pearson_p': stats_dict['pearson_p'],
                        'spearman_r': stats_dict['spearman_r'],
                        'spearman_p': stats_dict['spearman_p'],
                        'strength': self._correlation_strength(abs(stats_dict['pearson_r']))
                    })
        
        self.significant_df = pd.DataFrame(significant)
        if len(self.significant_df) > 0:
            self.significant_df = self.significant_df.sort_values('pearson_p')
        
        print(f"✓ Found {len(self.significant_df)} significant correlations")
        return self.significant_df
    
    def _correlation_strength(self, r):
        """Classify correlation strength"""
        r = abs(r)
        if r < 0.3:
            return 'weak'
        elif r < 0.5:
            return 'moderate'
        elif r < 0.7:
            return 'strong'
        else:
            return 'very_strong'
    
    def save_results(self, output_dir='results/summary_correlation'):
        """Save all correlation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        
        # 1. Save master dataset
        self.master_df.to_csv(output_path / 'master_dataset.csv', index=False)
        print("  ✓ Saved master_dataset.csv")
        
        # 2. Save all correlations
        all_corr = []
        for personality, summary_dict in self.correlations.items():
            for summary_metric, stats_dict in summary_dict.items():
                all_corr.append({
                    'personality_trait': personality,
                    'summary_metric': summary_metric,
                    'pearson_r': stats_dict['pearson_r'],
                    'pearson_p': stats_dict['pearson_p'],
                    'spearman_r': stats_dict['spearman_r'],
                    'spearman_p': stats_dict['spearman_p']
                })
        
        all_corr_df = pd.DataFrame(all_corr)
        all_corr_df = all_corr_df.sort_values('pearson_p')
        all_corr_df.to_csv(output_path / 'all_correlations.csv', index=False)
        print("  ✓ Saved all_correlations.csv")
        
        # 3. Save significant correlations
        if len(self.significant_df) > 0:
            self.significant_df.to_csv(output_path / 'significant_correlations.csv', index=False)
            print("  ✓ Saved significant_correlations.csv")
        
        # 4. Generate text report
        self._generate_text_report(output_path)
        
        print("✓ All results saved")
    
    def _generate_text_report(self, output_path):
        """Generate comprehensive text report"""
        report_file = output_path / 'correlation_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PERSONALITY-TEXT SUMMARY CORRELATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Participants: {len(self.master_df)}\n")
            f.write(f"  - Similar viewpoint: {len(self.master_df[self.master_df['Group'] == 'similar'])}\n")
            f.write(f"  - Opposing viewpoint: {len(self.master_df[self.master_df['Group'] == 'opposing'])}\n")
            f.write(f"  - Neutral viewpoint: {len(self.master_df[self.master_df['Group'] == 'neutral'])}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("PERSONALITY TRAIT STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            for trait in self.personality_questions.keys():
                values = self.master_df[trait]
                f.write(f"{trait.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {values.mean():.2f}\n")
                f.write(f"  Std:  {values.std():.2f}\n")
                f.write(f"  Range: {values.min():.0f} - {values.max():.0f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("SUMMARY SCORE STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            summary_metrics = ['Overall_Percentage', 'Semantic_Similarity', 'Topic_Coverage', 
                             'Factual_Accuracy', 'Compression_Ratio']
            
            for metric in summary_metrics:
                values = self.master_df[metric]
                f.write(f"{metric.replace('_', ' ')}:\n")
                f.write(f"  Mean: {values.mean():.3f}\n")
                f.write(f"  Std:  {values.std():.3f}\n")
                f.write(f"  Range: {values.min():.3f} - {values.max():.3f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("SIGNIFICANT CORRELATIONS (p < 0.05)\n")
            f.write("-" * 80 + "\n\n")
            
            if len(self.significant_df) > 0:
                for _, row in self.significant_df.iterrows():
                    f.write(f"{row['personality_trait'].replace('_', ' ').title()} × {row['summary_metric'].replace('_', ' ')}\n")
                    f.write(f"  Pearson r:   {row['pearson_r']:.4f} (p = {row['pearson_p']:.6f})\n")
                    f.write(f"  Spearman r:  {row['spearman_r']:.4f} (p = {row['spearman_p']:.6f})\n")
                    f.write(f"  Strength: {row['strength']}\n\n")
            else:
                f.write("No significant correlations found (p < 0.05)\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("TRENDING CORRELATIONS (0.05 < p < 0.10)\n")
            f.write("-" * 80 + "\n\n")
            
            # Find trending correlations
            trending = []
            for personality, summary_dict in self.correlations.items():
                for summary_metric, stats_dict in summary_dict.items():
                    if 0.05 <= stats_dict['pearson_p'] < 0.10:
                        trending.append({
                            'personality_trait': personality,
                            'summary_metric': summary_metric,
                            'pearson_r': stats_dict['pearson_r'],
                            'pearson_p': stats_dict['pearson_p'],
                            'spearman_r': stats_dict['spearman_r'],
                            'spearman_p': stats_dict['spearman_p']
                        })
            
            trending_df = pd.DataFrame(trending)
            if len(trending_df) > 0:
                trending_df = trending_df.sort_values('pearson_p')
                for _, row in trending_df.iterrows():
                    f.write(f"{row['personality_trait'].replace('_', ' ').title()} × {row['summary_metric'].replace('_', ' ')}\n")
                    f.write(f"  Pearson r:   {row['pearson_r']:.4f} (p = {row['pearson_p']:.6f})\n")
                    f.write(f"  Spearman r:  {row['spearman_r']:.4f} (p = {row['spearman_p']:.6f})\n\n")
            else:
                f.write("No trending correlations found (0.05 < p < 0.10)\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Correlation Strength:\n")
            f.write("  |r| < 0.3: Weak\n")
            f.write("  0.3 ≤ |r| < 0.5: Moderate\n")
            f.write("  0.5 ≤ |r| < 0.7: Strong\n")
            f.write("  |r| ≥ 0.7: Very strong\n\n")
            
            f.write("P-value Significance:\n")
            f.write("  p < 0.05: Statistically significant\n")
            f.write("  p < 0.01: Highly significant\n")
            f.write("  p < 0.001: Very highly significant\n\n")
            
            f.write("Note: With n=15, statistical power is limited. Non-significant results\n")
            f.write("do not necessarily mean no relationship exists.\n")
        
        print("  ✓ Saved correlation_report.txt")
    
    def visualize_correlations(self, output_dir='results/summary_correlation'):
        """Create visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualizations...")
        
        # 1. Correlation heatmap
        self._plot_correlation_heatmap(output_path)
        
        # 2. Top correlations bar chart
        self._plot_top_correlations(output_path)
        
        # 3. Summary scores by condition
        self._plot_scores_by_condition(output_path)
        
        # 4. Scatter plots for significant correlations
        if len(self.significant_df) > 0:
            self._plot_scatter_significant(output_path)
        
        print("✓ All visualizations saved")
    
    def _plot_correlation_heatmap(self, output_path):
        """Create heatmap of all correlations"""
        personality_cols = list(self.personality_questions.keys())
        summary_cols = ['Overall_Percentage', 'Semantic_Similarity', 'Topic_Coverage', 
                       'Factual_Accuracy', 'Compression_Ratio']
        
        # Build correlation matrix
        corr_matrix = np.zeros((len(personality_cols), len(summary_cols)))
        
        for i, personality in enumerate(personality_cols):
            for j, summary_metric in enumerate(summary_cols):
                corr_matrix[i, j] = self.correlations[personality][summary_metric]['pearson_r']
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Clean labels
        personality_labels = [p.replace('_', ' ').title() for p in personality_cols]
        summary_labels = [s.replace('_', ' ') for s in summary_cols]
        
        sns.heatmap(corr_matrix, 
                   xticklabels=summary_labels, 
                   yticklabels=personality_labels,
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, 
                   vmax=1,
                   cbar_kws={'label': 'Pearson Correlation'},
                   linewidths=0.5)
        
        plt.title('Personality Traits × Text Summary Quality Correlation Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Summary Quality Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Personality Traits', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved correlation_heatmap.png")
    
    def _plot_top_correlations(self, output_path):
        """Plot top 10 correlations by absolute value"""
        all_corr = []
        for personality, summary_dict in self.correlations.items():
            for summary_metric, stats_dict in summary_dict.items():
                all_corr.append({
                    'pair': f"{personality.replace('_', ' ').title()} × {summary_metric.replace('_', ' ')}",
                    'r': stats_dict['pearson_r'],
                    'p': stats_dict['pearson_p']
                })
        
        all_corr_df = pd.DataFrame(all_corr)
        all_corr_df['abs_r'] = all_corr_df['r'].abs()
        top_corr = all_corr_df.nlargest(10, 'abs_r')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['green' if r > 0 else 'red' for r in top_corr['r']]
        
        bars = ax.barh(range(len(top_corr)), top_corr['r'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr['pair'], fontsize=10)
        ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Personality-Summary Quality Correlations\n(by absolute strength)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add p-values
        for i, (_, row) in enumerate(top_corr.iterrows()):
            x_pos = row['r'] + (0.03 if row['r'] > 0 else -0.03)
            ha = 'left' if row['r'] > 0 else 'right'
            sig = '**' if row['p'] < 0.05 else ('*' if row['p'] < 0.10 else '')
            ax.text(x_pos, i, f"p={row['p']:.3f}{sig}", 
                   va='center', ha=ha, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved top_correlations.png")
    
    def _plot_scores_by_condition(self, output_path):
        """Plot summary scores by experimental condition"""
        summary_metrics = ['Overall_Percentage', 'Semantic_Similarity', 'Topic_Coverage', 
                          'Factual_Accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(summary_metrics):
            ax = axes[idx]
            
            # Group by condition
            condition_data = [
                self.master_df[self.master_df['Group'] == 'similar'][metric],
                self.master_df[self.master_df['Group'] == 'opposing'][metric],
                self.master_df[self.master_df['Group'] == 'neutral'][metric]
            ]
            
            positions = [1, 2, 3]
            bp = ax.boxplot(condition_data, positions=positions, widths=0.6, patch_artist=True,
                           labels=['Similar', 'Opposing', 'Neutral'])
            
            # Color boxes
            colors = ['#3498db', '#e74c3c', '#95a5a6']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(metric.replace('_', ' '), fontsize=11, fontweight='bold')
            ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add means
            means = [np.mean(data) for data in condition_data]
            ax.plot(positions, means, 'D', color='black', markersize=8, label='Mean', zorder=3)
            
            ax.legend(loc='upper right')
        
        plt.suptitle('Text Summary Quality by Experimental Condition', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path / 'scores_by_condition.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved scores_by_condition.png")
    
    def _plot_scatter_significant(self, output_path):
        """Create scatter plots for significant correlations"""
        n_sig = len(self.significant_df)
        
        if n_sig == 0:
            return
        
        # Calculate grid size
        n_cols = min(3, n_sig)
        n_rows = (n_sig + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_sig == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, (_, row) in enumerate(self.significant_df.iterrows()):
            ax = axes[idx]
            
            x = self.master_df[row['personality_trait']]
            y = self.master_df[row['summary_metric']]
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=100, edgecolors='black', linewidth=1)
            
            # Regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # Labels
            ax.set_xlabel(row['personality_trait'].replace('_', ' ').title(), 
                         fontsize=11, fontweight='bold')
            ax.set_ylabel(row['summary_metric'].replace('_', ' '), 
                         fontsize=11, fontweight='bold')
            
            # Title with stats
            title = f"r = {row['pearson_r']:.3f}, p = {row['pearson_p']:.4f}"
            ax.set_title(title, fontsize=10)
            
            ax.grid(alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_sig, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Significant Personality-Summary Quality Correlations', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path / 'scatter_significant.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved scatter_significant.png")


def main():
    """Run the personality-summary correlation analysis"""
    print("=" * 80)
    print("PERSONALITY-TEXT SUMMARY CORRELATION ANALYSIS")
    print("=" * 80)
    print()
    
    # Set up paths
    forms_csv = "forms_responses/Multimodal Analysis of Empathy in Opposing View Dialogues (Responses) - Form Responses 1.csv"
    grading_csv = "../text summarization 2/grading_results.csv"
    output_dir = "results/summary_correlation"
    
    # Initialize analyzer
    analyzer = PersonalitySummaryAnalyzer(forms_csv, grading_csv)
    
    # Load data
    analyzer.load_forms_data()
    analyzer.load_grading_data()
    
    # Create master dataset
    analyzer.create_master_dataset()
    
    # Calculate correlations
    analyzer.calculate_correlations()
    
    # Find significant correlations
    analyzer.find_significant_correlations(alpha=0.05)
    
    # Save results
    analyzer.save_results(output_dir)
    
    # Create visualizations
    analyzer.visualize_correlations(output_dir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nFiles generated:")
    print("  - master_dataset.csv (merged personality + summary data)")
    print("  - all_correlations.csv (all 40 correlation tests)")
    print("  - significant_correlations.csv (p < 0.05)")
    print("  - correlation_report.txt (detailed text report)")
    print("  - correlation_heatmap.png (visual matrix)")
    print("  - top_correlations.png (top 10 by strength)")
    print("  - scores_by_condition.png (boxplots by condition)")
    print("  - scatter_significant.png (scatter plots if significant)")
    print()


if __name__ == "__main__":
    main()
