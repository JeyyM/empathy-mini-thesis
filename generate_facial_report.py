"""
Comprehensive Facial Emotion Report Generator
Visualizes ALL facial emotion data collected
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

class ComprehensiveFacialReport:
    """Generate detailed reports showing ALL facial emotion data"""
    
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.dimensions = ['arousal', 'valence', 'intensity', 'excitement', 'calmness', 'positivity', 'negativity']
        
    def generate_report(self, df, save_path="facial_comprehensive_report.png"):
        """
        Generate comprehensive facial emotion report with ALL data visualized
        
        Args:
            df: DataFrame with facial emotion data
            save_path: Path to save the report image
        """
        # Detect column prefixes (facial_ or no prefix)
        has_prefix = 'facial_happy' in df.columns
        prefix = 'facial_' if has_prefix else ''
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
        
        fig.suptitle('🎭 COMPREHENSIVE FACIAL EMOTION ANALYSIS - ALL DATA', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ ROW 1: Basic Emotions Over Time ============
        
        # Plot 1: All 7 basic emotions
        ax1 = fig.add_subplot(gs[0, :2])
        colors = ['#FF4444', '#AA4444', '#FF8844', '#44FF44', '#4444FF', '#FF44FF', '#888888']
        for emotion, color in zip(self.emotions, colors):
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                ax1.plot(time_data, df[col_name], label=emotion.capitalize(), 
                        color=color, linewidth=2, alpha=0.8)
        ax1.set_title('📊 All 7 Basic Emotions Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Emotion Intensity (0-1)')
        ax1.legend(loc='upper right', ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # Plot 2: Emotion intensity heatmap
        ax2 = fig.add_subplot(gs[0, 2:])
        emotion_matrix = []
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                emotion_matrix.append(df[col_name].values)
        
        if emotion_matrix:
            im = ax2.imshow(emotion_matrix, aspect='auto', cmap='YlOrRd', 
                           interpolation='nearest', extent=[time_data.min(), time_data.max(), 0, len(self.emotions)])
            ax2.set_yticks(np.arange(len(self.emotions)) + 0.5)
            ax2.set_yticklabels([e.capitalize() for e in self.emotions])
            ax2.set_xlabel('Time (seconds)')
            ax2.set_title('🔥 Emotion Intensity Heatmap', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax2, label='Intensity')
        
        # ============ ROW 2: Dimensional Analysis ============
        
        # Plot 3: Arousal and Valence
        ax3 = fig.add_subplot(gs[1, 0])
        arousal_col = f'{prefix}arousal'
        valence_col = f'{prefix}valence'
        if arousal_col in df.columns and valence_col in df.columns:
            ax3.plot(time_data, df[arousal_col], label='Arousal (Energy)', 
                    color='#FF6B6B', linewidth=2.5)
            ax3.plot(time_data, df[valence_col], label='Valence (Mood)', 
                    color='#4ECDC4', linewidth=2.5)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_title('⚡ Arousal & Valence', fontsize=13, fontweight='bold')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Level')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(-1.1, 1.1)
        
        # Plot 4: Intensity
        ax4 = fig.add_subplot(gs[1, 1])
        intensity_col = f'{prefix}intensity'
        if intensity_col in df.columns:
            ax4.fill_between(time_data, 0, df[intensity_col], alpha=0.6, color='#FF8C00')
            ax4.plot(time_data, df[intensity_col], color='#FF4500', linewidth=2)
            ax4.set_title('💥 Emotional Intensity', fontsize=13, fontweight='bold')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Intensity')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.05)
        
        # Plot 5: Excitement vs Calmness
        ax5 = fig.add_subplot(gs[1, 2])
        excitement_col = f'{prefix}excitement'
        calmness_col = f'{prefix}calmness'
        if excitement_col in df.columns and calmness_col in df.columns:
            ax5.plot(time_data, df[excitement_col], label='Excitement', 
                    color='#FF1744', linewidth=2)
            ax5.plot(time_data, df[calmness_col], label='Calmness', 
                    color='#00BFA5', linewidth=2)
            ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax5.set_title('🎢 Excitement vs Calmness', fontsize=13, fontweight='bold')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Level')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Positivity vs Negativity
        ax6 = fig.add_subplot(gs[1, 3])
        positivity_col = f'{prefix}positivity'
        negativity_col = f'{prefix}negativity'
        if positivity_col in df.columns and negativity_col in df.columns:
            ax6.plot(time_data, df[positivity_col], label='Positivity', 
                    color='#76FF03', linewidth=2)
            ax6.plot(time_data, df[negativity_col], label='Negativity', 
                    color='#D32F2F', linewidth=2)
            ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax6.set_title('😊😠 Positivity vs Negativity', fontsize=13, fontweight='bold')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Level')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # ============ ROW 3: Distribution Analysis ============
        
        # Plot 7: Emotion distribution (violin plots)
        ax7 = fig.add_subplot(gs[2, :2])
        emotion_data_for_violin = []
        emotion_labels = []
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                emotion_data_for_violin.append(df[col_name].values)
                emotion_labels.append(emotion.capitalize())
        
        if emotion_data_for_violin:
            parts = ax7.violinplot(emotion_data_for_violin, positions=range(len(emotion_labels)),
                                   showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('#FF6B6B')
                pc.set_alpha(0.7)
            ax7.set_xticks(range(len(emotion_labels)))
            ax7.set_xticklabels(emotion_labels, rotation=45)
            ax7.set_title('🎻 Emotion Distribution (Violin Plots)', fontsize=14, fontweight='bold')
            ax7.set_ylabel('Intensity')
            ax7.grid(True, alpha=0.3, axis='y')
        
        # Plot 8: Quadrant distribution
        ax8 = fig.add_subplot(gs[2, 2:])
        quadrant_col = f'{prefix}quadrant'
        if quadrant_col in df.columns:
            quadrant_counts = df[quadrant_col].value_counts()
            colors_quad = {'Excited': '#FFD700', 'Agitated': '#FF6347', 
                          'Calm': '#90EE90', 'Depressed': '#87CEEB'}
            bars = ax8.bar(range(len(quadrant_counts)), quadrant_counts.values,
                          color=[colors_quad.get(q, '#888888') for q in quadrant_counts.index])
            ax8.set_xticks(range(len(quadrant_counts)))
            ax8.set_xticklabels(quadrant_counts.index, rotation=45)
            ax8.set_title('🧭 Emotional Quadrant Distribution', fontsize=14, fontweight='bold')
            ax8.set_ylabel('Frequency')
            
            # Add percentages on bars
            for i, (bar, count) in enumerate(zip(bars, quadrant_counts.values)):
                percentage = (count / len(df)) * 100
                ax8.text(i, count + 0.5, f'{percentage:.1f}%', 
                        ha='center', fontsize=10, fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='y')
        
        # ============ ROW 4: Statistical Summary & Correlations ============
        
        # Plot 9: Statistical summary table
        ax9 = fig.add_subplot(gs[3, :2])
        ax9.axis('off')
        
        # Calculate statistics
        stats_data = []
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                stats_data.append([
                    emotion.capitalize(),
                    f"{df[col_name].mean():.3f}",
                    f"{df[col_name].std():.3f}",
                    f"{df[col_name].min():.3f}",
                    f"{df[col_name].max():.3f}",
                    f"{df[col_name].median():.3f}"
                ])
        
        if stats_data:
            table = ax9.table(cellText=stats_data,
                            colLabels=['Emotion', 'Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style header
            for i in range(6):
                table[(0, i)].set_facecolor('#4ECDC4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(stats_data) + 1):
                for j in range(6):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#F0F0F0')
        
        ax9.set_title('📈 Statistical Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Plot 10: Correlation heatmap
        ax10 = fig.add_subplot(gs[3, 2:])
        
        # Get all numeric columns for correlation
        numeric_cols = []
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                numeric_cols.append(col_name)
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            im = ax10.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax10.set_xticks(range(len(numeric_cols)))
            ax10.set_yticks(range(len(numeric_cols)))
            ax10.set_xticklabels([c.replace(prefix, '').capitalize() for c in numeric_cols], rotation=45, ha='right')
            ax10.set_yticklabels([c.replace(prefix, '').capitalize() for c in numeric_cols])
            ax10.set_title('🔗 Emotion Correlations', fontsize=14, fontweight='bold')
            
            # Add correlation values
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    text = ax10.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=ax10, label='Correlation')
        
        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Comprehensive facial report saved to: {save_path}")
        plt.close()
        
        # Print summary statistics
        self.print_summary(df, prefix)
    
    def print_summary(self, df, prefix=''):
        """Print detailed summary statistics"""
        print(f"\n{'='*80}")
        print("📊 FACIAL EMOTION ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Total samples: {len(df)}")
        
        if 'time_seconds' in df.columns:
            duration = df['time_seconds'].max() - df['time_seconds'].min()
            print(f"Duration: {duration:.1f} seconds")
        
        print(f"\n--- Emotion Averages ---")
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                avg = df[col_name].mean()
                print(f"{emotion.capitalize():>10s}: {avg:.3f}")
        
        print(f"\n--- Dominant Emotions ---")
        emotion_avgs = {}
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                emotion_avgs[emotion] = df[col_name].mean()
        
        sorted_emotions = sorted(emotion_avgs.items(), key=lambda x: x[1], reverse=True)
        for i, (emotion, avg) in enumerate(sorted_emotions[:3], 1):
            print(f"{i}. {emotion.capitalize()}: {avg:.3f}")
        
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Test with sample data
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "webcam_live_emotion_data.csv"
    
    try:
        df = pd.read_csv(csv_file)
        reporter = ComprehensiveFacialReport()
        reporter.generate_report(df, save_path="facial_comprehensive_report.png")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("Usage: python generate_facial_report.py [csv_file]")
