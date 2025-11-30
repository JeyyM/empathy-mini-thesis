"""
Personality-Empathy Correlation Analysis
Connects participants' self-reported communication personality traits with their empathy summary scores
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

class PersonalityEmpathyAnalyzer:
    def __init__(self, forms_csv_path, results_base_path):
        """
        Initialize analyzer with paths to data
        
        Args:
            forms_csv_path: Path to Google Forms responses CSV
            results_base_path: Base path to results directory containing participant folders
        """
        self.forms_csv_path = forms_csv_path
        self.results_base_path = Path(results_base_path)
        
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
        
        # Email to name mapping based on file structure
        self.email_to_name_map = {}
        
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
        
        # Clean email addresses
        self.forms_df['email'] = self.forms_df['email'].str.strip().str.lower()
        
        print(f"✓ Loaded {len(self.forms_df)} form responses")
        print(f"  Participants: {', '.join(self.forms_df['full_name'].tolist())}")
        return self.forms_df
    
    def build_name_mapping(self):
        """Build mapping between emails and folder names"""
        print("\nBuilding email-to-folder mapping...")
        
        # Manual mapping for cases where automated matching fails
        manual_map = {
            'EthanOng': 'Ethan Brook Ong',
            'MaggieOng': 'Maggie Chong',
            'KeithziCantona': 'Keithzi Rhaz Cantona',
            'AndreMarco': 'Pete Andre Marco',
            'SeanTe': 'Sean Te',
            'AaronDionisio': 'Aaron Dionisio',
            'ArianPates': 'Charlz Arian S. Pates',
            'EthanPlaza': 'Ethan Luise Benedict C. Plaza',
            'MarwahMuti': 'Marwah B. Muti',
            'SamuelLim': 'Samuel Lim',
            'MiguelBorromeo': 'Anton Miguel G. Borromeo',
            'MiguelNg': 'Miguel Zakia Ng',
            'RandellFabico': 'Louise Randell-so R. Fabico',
            'RusselGalan': 'Russell Emmanuel G. Galan',
            'RyanSo': 'Ryan Justin So'
        }
        
        # Scan all subdirectories for participant folders
        for condition_dir in ['Similar', 'Opposing', 'Neutral']:
            condition_path = self.results_base_path / condition_dir
            if not condition_path.exists():
                continue
                
            for participant_folder in condition_path.iterdir():
                if participant_folder.is_dir():
                    folder_name = participant_folder.name
                    
                    # Try manual mapping first
                    target_name = manual_map.get(folder_name)
                    
                    if target_name:
                        # Find the corresponding row in forms_df
                        matching_row = self.forms_df[self.forms_df['full_name'] == target_name]
                        if not matching_row.empty:
                            row = matching_row.iloc[0]
                            self.email_to_name_map[row['email']] = {
                                'folder_name': folder_name,
                                'condition': condition_dir,
                                'full_name': row['full_name'],
                                'folder_path': participant_folder
                            }
                            print(f"  ✓ {row['full_name']} ({row['email']}) → {condition_dir}/{folder_name}")
                            continue
                    
                    # Fall back to fuzzy matching
                    for idx, row in self.forms_df.iterrows():
                        if row['email'] in self.email_to_name_map:
                            continue
                            
                        full_name = row['full_name']
                        # Remove spaces and special characters for matching
                        name_normalized = ''.join(full_name.split()).lower()
                        folder_normalized = folder_name.lower()
                        
                        if name_normalized in folder_normalized or folder_normalized in name_normalized:
                            self.email_to_name_map[row['email']] = {
                                'folder_name': folder_name,
                                'condition': condition_dir,
                                'full_name': full_name,
                                'folder_path': participant_folder
                            }
                            print(f"  ✓ {full_name} ({row['email']}) → {condition_dir}/{folder_name}")
                            break
        
        print(f"\nMapped {len(self.email_to_name_map)} participants to their result folders")
        return self.email_to_name_map
    
    def load_participant_summary_scores(self, participant_info):
        """Load and summarize emotion data for a participant"""
        folder_path = participant_info['folder_path']
        csv_files = list(folder_path.glob('*_ml_fusion.csv'))
        
        if not csv_files:
            print(f"  ⚠ No fusion CSV found for {participant_info['full_name']}")
            return None
        
        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)
        
        # Calculate summary statistics
        summary = {
            # Arousal and valence
            'mean_arousal': df['fused_arousal'].mean(),
            'std_arousal': df['fused_arousal'].std(),
            'mean_valence': df['fused_valence'].mean(),
            'std_valence': df['fused_valence'].std(),
            
            # Intensity and stress
            'mean_intensity': df['fused_intensity'].mean(),
            'mean_stress': df['fused_stress'].mean(),
            
            # Positivity and negativity
            'mean_positivity': df['fused_positivity'].mean(),
            'mean_negativity': df['fused_negativity'].mean(),
            
            # Excitement and calmness
            'mean_excitement': df['fused_excitement'].mean(),
            'mean_calmness': df['fused_calmness'].mean(),
            
            # Individual emotions (proportions)
            'prop_angry': df['fused_angry'].mean(),
            'prop_disgust': df['fused_disgust'].mean(),
            'prop_fear': df['fused_fear'].mean(),
            'prop_happy': df['fused_happy'].mean(),
            'prop_sad': df['fused_sad'].mean(),
            'prop_surprise': df['fused_surprise'].mean(),
            'prop_neutral': df['fused_neutral'].mean(),
            
            # Variability (emotional stability)
            'arousal_variability': df['fused_arousal'].std(),
            'valence_variability': df['fused_valence'].std(),
            'intensity_variability': df['fused_intensity'].std(),
            
            # Quadrant distribution
            'prop_excited': (df['fused_quadrant'] == 'Excited').mean(),
            'prop_stressed': (df['fused_quadrant'] == 'Stressed').mean(),
            'prop_calm': (df['fused_quadrant'] == 'Calm').mean(),
            'prop_tired': (df['fused_quadrant'] == 'Tired').mean(),
        }
        
        return summary
    
    def create_master_dataset(self):
        """Combine personality traits with emotion summary scores"""
        print("\nCreating master dataset...")
        
        master_data = []
        
        for idx, row in self.forms_df.iterrows():
            email = row['email']
            
            if email not in self.email_to_name_map:
                print(f"  ⚠ No results folder found for {row['full_name']} ({email})")
                continue
            
            participant_info = self.email_to_name_map[email]
            summary_scores = self.load_participant_summary_scores(participant_info)
            
            if summary_scores is None:
                continue
            
            # Combine personality traits with summary scores
            participant_data = {
                'email': email,
                'full_name': row['full_name'],
                'condition': participant_info['condition'],
                **{k: row[k] for k in self.personality_questions.keys()},
                **summary_scores
            }
            
            master_data.append(participant_data)
            print(f"  ✓ Processed {row['full_name']}")
        
        self.master_df = pd.DataFrame(master_data)
        print(f"\n✓ Master dataset created with {len(self.master_df)} participants")
        
        return self.master_df
    
    def calculate_correlations(self):
        """Calculate correlations between personality traits and emotion metrics"""
        print("\nCalculating correlations...")
        
        personality_cols = list(self.personality_questions.keys())
        emotion_cols = [col for col in self.master_df.columns 
                       if col.startswith(('mean_', 'std_', 'prop_', 'arousal_', 'valence_', 'intensity_'))]
        
        # Calculate Pearson and Spearman correlations
        correlations = {}
        
        for personality in personality_cols:
            correlations[personality] = {}
            for emotion in emotion_cols:
                # Pearson correlation
                pearson_r, pearson_p = stats.pearsonr(
                    self.master_df[personality], 
                    self.master_df[emotion]
                )
                
                # Spearman correlation (for ordinal data)
                spearman_r, spearman_p = stats.spearmanr(
                    self.master_df[personality], 
                    self.master_df[emotion]
                )
                
                correlations[personality][emotion] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
        
        self.correlations = correlations
        return correlations
    
    def find_significant_correlations(self, alpha=0.05):
        """Find statistically significant correlations"""
        print(f"\nFinding significant correlations (α = {alpha})...")
        
        significant = []
        
        for personality, emotion_dict in self.correlations.items():
            for emotion, stats_dict in emotion_dict.items():
                if stats_dict['pearson_p'] < alpha:
                    significant.append({
                        'personality_trait': personality,
                        'emotion_metric': emotion,
                        'pearson_r': stats_dict['pearson_r'],
                        'pearson_p': stats_dict['pearson_p'],
                        'spearman_r': stats_dict['spearman_r'],
                        'spearman_p': stats_dict['spearman_p'],
                        'strength': self._correlation_strength(abs(stats_dict['pearson_r']))
                    })
        
        self.significant_df = pd.DataFrame(significant)
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
    
    def visualize_correlations(self, output_dir='results/correlation_analysis'):
        """Create comprehensive visualization of correlations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating visualizations in {output_dir}...")
        
        # 1. Heatmap of all correlations
        self._plot_correlation_heatmap(output_path)
        
        # 2. Top significant correlations
        self._plot_top_correlations(output_path)
        
        # 3. Scatter plots for significant relationships
        self._plot_scatter_significant(output_path)
        
        # 4. Personality profile comparison by condition
        self._plot_personality_by_condition(output_path)
        
        # 5. Emotion metrics comparison by condition
        self._plot_emotions_by_condition(output_path)
        
        print("✓ All visualizations saved")
    
    def _plot_correlation_heatmap(self, output_path):
        """Create heatmap of personality-emotion correlations"""
        personality_cols = list(self.personality_questions.keys())
        emotion_cols = [col for col in self.master_df.columns 
                       if col.startswith(('mean_', 'prop_'))]
        
        # Build correlation matrix
        corr_matrix = np.zeros((len(personality_cols), len(emotion_cols)))
        
        for i, personality in enumerate(personality_cols):
            for j, emotion in enumerate(emotion_cols):
                corr_matrix[i, j] = self.correlations[personality][emotion]['pearson_r']
        
        # Plot
        plt.figure(figsize=(20, 10))
        sns.heatmap(corr_matrix, 
                   xticklabels=emotion_cols, 
                   yticklabels=personality_cols,
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, 
                   vmax=1,
                   cbar_kws={'label': 'Pearson Correlation'})
        
        plt.title('Personality Traits vs Emotion Metrics Correlation Heatmap', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Emotion Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Personality Traits', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved correlation_heatmap.png")
    
    def _plot_top_correlations(self, output_path):
        """Plot top 15 significant correlations"""
        if len(self.significant_df) == 0:
            print("  ⚠ No significant correlations to plot")
            return
        
        top_n = min(15, len(self.significant_df))
        top_corr = self.significant_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create labels
        labels = [f"{row['personality_trait']} × {row['emotion_metric']}" 
                 for _, row in top_corr.iterrows()]
        
        # Create bar colors based on positive/negative correlation
        colors = ['green' if r > 0 else 'red' for r in top_corr['pearson_r']]
        
        # Plot
        bars = ax.barh(range(len(labels)), top_corr['pearson_r'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Significant Personality-Emotion Correlations', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add p-values as text
        for i, (_, row) in enumerate(top_corr.iterrows()):
            x_pos = row['pearson_r'] + (0.02 if row['pearson_r'] > 0 else -0.02)
            ha = 'left' if row['pearson_r'] > 0 else 'right'
            ax.text(x_pos, i, f"p={row['pearson_p']:.4f}", 
                   va='center', ha=ha, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved top_correlations.png")
    
    def _plot_scatter_significant(self, output_path, top_n=6):
        """Create scatter plots for top significant correlations"""
        if len(self.significant_df) == 0:
            return
        
        top_corr = self.significant_df.head(top_n)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(top_corr.iterrows()):
            if idx >= 6:
                break
                
            personality = row['personality_trait']
            emotion = row['emotion_metric']
            
            ax = axes[idx]
            
            # Create scatter plot
            scatter = ax.scatter(self.master_df[personality], 
                               self.master_df[emotion],
                               c=self.master_df['condition'].map({
                                   'Similar': 'blue', 
                                   'Opposing': 'red', 
                                   'Neutral': 'green'
                               }),
                               s=100, 
                               alpha=0.6,
                               edgecolors='black',
                               linewidth=1)
            
            # Add trend line
            z = np.polyfit(self.master_df[personality], self.master_df[emotion], 1)
            p = np.poly1d(z)
            x_line = np.linspace(self.master_df[personality].min(), 
                               self.master_df[personality].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            # Labels
            ax.set_xlabel(personality.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_ylabel(emotion.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            ax.set_title(f"r={row['pearson_r']:.3f}, p={row['pearson_p']:.4f}", 
                        fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.6, label='Similar'),
            Patch(facecolor='red', alpha=0.6, label='Opposing'),
            Patch(facecolor='green', alpha=0.6, label='Neutral')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.suptitle('Top Personality-Emotion Correlations (by Condition)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path / 'scatter_significant_correlations.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved scatter_significant_correlations.png")
    
    def _plot_personality_by_condition(self, output_path):
        """Compare personality traits across conditions"""
        personality_cols = list(self.personality_questions.keys())
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, trait in enumerate(personality_cols):
            ax = axes[idx]
            
            data_by_condition = [
                self.master_df[self.master_df['condition'] == cond][trait].values
                for cond in ['Similar', 'Opposing', 'Neutral']
            ]
            
            bp = ax.boxplot(data_by_condition, 
                          labels=['Similar', 'Opposing', 'Neutral'],
                          patch_artist=True,
                          showmeans=True)
            
            # Color boxes
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Score (1-5)', fontsize=10)
            ax.set_title(trait.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 6)
        
        plt.suptitle('Personality Traits Distribution by Condition', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'personality_by_condition.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved personality_by_condition.png")
    
    def _plot_emotions_by_condition(self, output_path):
        """Compare emotion metrics across conditions"""
        emotion_metrics = ['mean_arousal', 'mean_valence', 'mean_intensity', 
                          'mean_stress', 'mean_positivity', 'mean_negativity',
                          'mean_excitement', 'mean_calmness']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(emotion_metrics):
            ax = axes[idx]
            
            data_by_condition = [
                self.master_df[self.master_df['condition'] == cond][metric].values
                for cond in ['Similar', 'Opposing', 'Neutral']
            ]
            
            bp = ax.boxplot(data_by_condition, 
                          labels=['Similar', 'Opposing', 'Neutral'],
                          patch_artist=True,
                          showmeans=True)
            
            # Color boxes
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Emotion Metrics Distribution by Condition', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'emotions_by_condition.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Saved emotions_by_condition.png")
    
    def generate_report(self, output_path='results/correlation_analysis/correlation_report.txt'):
        """Generate comprehensive text report"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PERSONALITY-EMPATHY CORRELATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Participants: {len(self.master_df)}\n")
            f.write(f"  - Similar viewpoint: {(self.master_df['condition'] == 'Similar').sum()}\n")
            f.write(f"  - Opposing viewpoint: {(self.master_df['condition'] == 'Opposing').sum()}\n")
            f.write(f"  - Neutral viewpoint: {(self.master_df['condition'] == 'Neutral').sum()}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("PERSONALITY TRAIT STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            personality_cols = list(self.personality_questions.keys())
            for trait in personality_cols:
                f.write(f"{trait.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {self.master_df[trait].mean():.2f}\n")
                f.write(f"  Std:  {self.master_df[trait].std():.2f}\n")
                f.write(f"  Range: {self.master_df[trait].min():.0f} - {self.master_df[trait].max():.0f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("SIGNIFICANT CORRELATIONS (p < 0.05)\n")
            f.write("-" * 80 + "\n\n")
            
            if len(self.significant_df) > 0:
                for _, row in self.significant_df.iterrows():
                    f.write(f"{row['personality_trait'].replace('_', ' ').title()} × "
                           f"{row['emotion_metric'].replace('_', ' ').title()}\n")
                    f.write(f"  Pearson r:  {row['pearson_r']:>7.4f} (p = {row['pearson_p']:.6f})\n")
                    f.write(f"  Spearman r: {row['spearman_r']:>7.4f} (p = {row['spearman_p']:.6f})\n")
                    f.write(f"  Strength: {row['strength']}\n\n")
            else:
                f.write("No statistically significant correlations found (p < 0.05)\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("HOW CORRELATION WORKS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("""
Correlation Explanation:
------------------------

1. **Pearson Correlation Coefficient (r)**:
   - Measures LINEAR relationship between two variables
   - Range: -1 to +1
     * +1: Perfect positive linear relationship
     * -1: Perfect negative linear relationship
     *  0: No linear relationship
   - Interpretation:
     * |r| < 0.3: Weak correlation
     * 0.3 ≤ |r| < 0.5: Moderate correlation
     * 0.5 ≤ |r| < 0.7: Strong correlation
     * |r| ≥ 0.7: Very strong correlation

2. **Spearman Correlation Coefficient (ρ)**:
   - Measures MONOTONIC relationship (not necessarily linear)
   - Better for ordinal data (like Likert scales 1-5)
   - Robust to outliers
   - Same interpretation as Pearson

3. **P-value**:
   - Probability that the observed correlation occurred by chance
   - p < 0.05: Statistically significant (reject null hypothesis)
   - p < 0.01: Highly significant
   - p < 0.001: Very highly significant

4. **What This Analysis Shows**:
   - Which personality traits predict emotional responses during conversations
   - Whether certain traits lead to more positive/negative emotions
   - How communication style affects emotional arousal and intensity
   - Differences between Similar, Opposing, and Neutral viewpoint conditions

5. **Example Interpretation**:
   If "stays_calm × mean_stress" has r = -0.65, p = 0.001:
   - Strong negative correlation
   - People who report staying calm show LESS stress in conversations
   - This relationship is statistically significant
   - The pattern is consistent and unlikely due to chance

6. **Limitations**:
   - Correlation ≠ Causation
   - Small sample size may limit statistical power
   - Self-reported personality may differ from actual behavior
   - Context-dependent (specific to this study's conversation topics)
""")
        
        print(f"\n✓ Report saved to {output_file}")
    
    def save_master_dataset(self, output_path='results/correlation_analysis/master_dataset.csv'):
        """Save the combined dataset"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.master_df.to_csv(output_file, index=False)
        print(f"✓ Master dataset saved to {output_file}")
    
    def save_correlations(self, output_path='results/correlation_analysis/all_correlations.csv'):
        """Save all correlations to CSV"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for personality, emotion_dict in self.correlations.items():
            for emotion, stats_dict in emotion_dict.items():
                rows.append({
                    'personality_trait': personality,
                    'emotion_metric': emotion,
                    'pearson_r': stats_dict['pearson_r'],
                    'pearson_p': stats_dict['pearson_p'],
                    'spearman_r': stats_dict['spearman_r'],
                    'spearman_p': stats_dict['spearman_p']
                })
        
        corr_df = pd.DataFrame(rows)
        corr_df = corr_df.sort_values('pearson_p')
        corr_df.to_csv(output_file, index=False)
        
        print(f"✓ All correlations saved to {output_file}")


def main():
    """Run complete analysis pipeline"""
    print("=" * 80)
    print("PERSONALITY-EMPATHY CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Initialize analyzer
    forms_csv = r"c:\Users\chris\empathy-mini-thesis\results\forms_responses\Multimodal Analysis of Empathy in Opposing View Dialogues (Responses) - Form Responses 1.csv"
    results_base = r"c:\Users\chris\empathy-mini-thesis\results"
    
    analyzer = PersonalityEmpathyAnalyzer(forms_csv, results_base)
    
    # Run analysis pipeline
    analyzer.load_forms_data()
    analyzer.build_name_mapping()
    analyzer.create_master_dataset()
    analyzer.calculate_correlations()
    analyzer.find_significant_correlations(alpha=0.05)
    
    # Generate outputs
    analyzer.save_master_dataset()
    analyzer.save_correlations()
    analyzer.visualize_correlations()
    analyzer.generate_report()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files in 'results/correlation_analysis/':")
    print("  1. master_dataset.csv - Combined personality + emotion data")
    print("  2. all_correlations.csv - All correlation coefficients")
    print("  3. correlation_heatmap.png - Visual matrix of all correlations")
    print("  4. top_correlations.png - Bar chart of strongest relationships")
    print("  5. scatter_significant_correlations.png - Scatter plots with trendlines")
    print("  6. personality_by_condition.png - Personality trait distributions")
    print("  7. emotions_by_condition.png - Emotion metric distributions")
    print("  8. correlation_report.txt - Detailed text report with explanations")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
