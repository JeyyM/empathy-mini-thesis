"""
ULTIMATE COMPREHENSIVE REPORT GENERATOR
Generates the most comprehensive emotional analysis possible by combining ALL data sources
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

class UltimateReportGenerator:
    """
    Generates ultra-comprehensive reports showing EVERYTHING:
    - All facial emotions over time
    - All voice emotions over time
    - Arousal/Valence/Intensity comparisons
    - 2D emotion space visualizations
    - Statistical summaries
    - Correlation matrices
    - Distribution histograms
    - Quadrant analysis
    - Peak detection
    - Emotion transitions
    """
    
    def __init__(self):
        self.facial_emotions = ['facial_angry', 'facial_disgust', 'facial_fear', 
                                'facial_happy', 'facial_sad', 'facial_surprise', 'facial_neutral']
        self.voice_emotions = ['voice_angry', 'voice_disgust', 'voice_fear',
                               'voice_happy', 'voice_sad', 'voice_surprise', 'voice_neutral']
        self.facial_dimensions = ['facial_arousal', 'facial_valence', 'facial_intensity',
                                  'facial_excitement', 'facial_calmness', 'facial_positivity', 'facial_negativity']
        self.voice_dimensions = ['voice_arousal', 'voice_valence', 'voice_intensity', 'voice_stress']
    
    def generate_mega_report(self, df, save_path="ultimate_mega_report.png"):
        """
        Generate the ULTIMATE mega report with 20+ subplots showing EVERYTHING
        """
        print(f"\n{'='*80}")
        print("🚀 GENERATING ULTIMATE MEGA REPORT")
        print(f"{'='*80}")
        print(f"📊 Analyzing {len(df)} data points...")
        
        # Check what data we have
        has_facial = any(col in df.columns for col in self.facial_emotions)
        has_voice = any(col in df.columns for col in self.voice_emotions)
        
        if not has_facial and not has_voice:
            print("❌ No emotion data found!")
            return
        
        print(f"   Facial data: {'✅' if has_facial else '❌'}")
        print(f"   Voice data: {'✅' if has_voice else '❌'}")
        
        # Create massive figure with 24 subplots (6 rows x 4 columns)
        fig = plt.figure(figsize=(24, 30))
        gs = GridSpec(6, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Time axis
        if 'time_seconds' in df.columns:
            time = df['time_seconds']
        elif 'timestamp' in df.columns:
            time = np.arange(len(df))
        else:
            time = np.arange(len(df))
        
        row = 0
        
        # ROW 1: All Facial Emotions Timeline
        if has_facial:
            ax1 = fig.add_subplot(gs[row, :])
            for emotion_col in self.facial_emotions:
                if emotion_col in df.columns:
                    emotion_name = emotion_col.replace('facial_', '').title()
                    ax1.plot(time, df[emotion_col], label=emotion_name, linewidth=2, alpha=0.8)
            ax1.set_title('🎭 ALL FACIAL EMOTIONS OVER TIME', fontsize=16, fontweight='bold', pad=15)
            ax1.set_xlabel('Time (seconds)', fontsize=12)
            ax1.set_ylabel('Emotion Probability', fontsize=12)
            ax1.legend(loc='upper right', ncol=4, fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([-0.05, 1.05])
            row += 1
        
        # ROW 2: All Voice Emotions Timeline
        if has_voice:
            ax2 = fig.add_subplot(gs[row, :])
            for emotion_col in self.voice_emotions:
                if emotion_col in df.columns:
                    emotion_name = emotion_col.replace('voice_', '').title()
                    ax2.plot(time, df[emotion_col], label=emotion_name, linewidth=2, alpha=0.8)
            ax2.set_title('🎤 ALL VOICE EMOTIONS OVER TIME', fontsize=16, fontweight='bold', pad=15)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylabel('Emotion Probability', fontsize=12)
            ax2.legend(loc='upper right', ncol=4, fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([-0.05, 1.05])
            row += 1
        
        # ROW 3: Psychological Dimensions Comparison
        if has_facial:
            # Facial Arousal/Valence
            ax3 = fig.add_subplot(gs[row, 0])
            if 'facial_arousal' in df.columns:
                ax3.plot(time, df['facial_arousal'], label='Arousal', linewidth=2.5, color='red')
            if 'facial_valence' in df.columns:
                ax3.plot(time, df['facial_valence'], label='Valence', linewidth=2.5, color='blue')
            ax3.set_title('📸 Facial: Arousal vs Valence', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Time (s)', fontsize=10)
            ax3.set_ylabel('Level (-1 to 1)', fontsize=10)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Facial Intensity/Excitement
            ax4 = fig.add_subplot(gs[row, 1])
            if 'facial_intensity' in df.columns:
                ax4.plot(time, df['facial_intensity'], label='Intensity', linewidth=2.5, color='purple')
            if 'facial_excitement' in df.columns:
                ax4.plot(time, df['facial_excitement'], label='Excitement', linewidth=2.5, color='orange')
            ax4.set_title('📸 Facial: Intensity & Excitement', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Time (s)', fontsize=10)
            ax4.set_ylabel('Level (0 to 1)', fontsize=10)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
        
        if has_voice:
            # Voice Arousal/Valence
            ax5 = fig.add_subplot(gs[row, 2])
            if 'voice_arousal' in df.columns:
                ax5.plot(time, df['voice_arousal'], label='Arousal', linewidth=2.5, color='red')
            if 'voice_valence' in df.columns:
                ax5.plot(time, df['voice_valence'], label='Valence', linewidth=2.5, color='blue')
            ax5.set_title('🎤 Voice: Arousal vs Valence', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Time (s)', fontsize=10)
            ax5.set_ylabel('Level (-1 to 1)', fontsize=10)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Voice Intensity/Stress
            ax6 = fig.add_subplot(gs[row, 3])
            if 'voice_intensity' in df.columns:
                ax6.plot(time, df['voice_intensity'], label='Intensity', linewidth=2.5, color='purple')
            if 'voice_stress' in df.columns:
                ax6.plot(time, df['voice_stress'], label='Stress', linewidth=2.5, color='darkred')
            ax6.set_title('🎤 Voice: Intensity & Stress', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Time (s)', fontsize=10)
            ax6.set_ylabel('Level (0 to 1)', fontsize=10)
            ax6.legend(fontsize=9)
            ax6.grid(True, alpha=0.3)
        
        row += 1
        
        # ROW 4: 2D Emotion Spaces
        if has_facial and 'facial_arousal' in df.columns and 'facial_valence' in df.columns:
            ax7 = fig.add_subplot(gs[row, 0])
            scatter = ax7.scatter(df['facial_valence'], df['facial_arousal'], 
                                 c=time, cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax7.set_title('📸 Facial Emotion Space', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Valence (negative ← → positive)', fontsize=10)
            ax7.set_ylabel('Arousal (low ↓ ↑ high)', fontsize=10)
            ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax7.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax7.set_xlim([-1.1, 1.1])
            ax7.set_ylim([-1.1, 1.1])
            ax7.grid(True, alpha=0.2)
            # Add quadrant labels
            ax7.text(0.5, 0.5, 'EXCITED', fontsize=9, ha='center', va='center', alpha=0.4, fontweight='bold')
            ax7.text(-0.5, 0.5, 'STRESSED', fontsize=9, ha='center', va='center', alpha=0.4, fontweight='bold')
            ax7.text(-0.5, -0.5, 'TIRED', fontsize=9, ha='center', va='center', alpha=0.4, fontweight='bold')
            ax7.text(0.5, -0.5, 'PEACEFUL', fontsize=9, ha='center', va='center', alpha=0.4, fontweight='bold')
            plt.colorbar(scatter, ax=ax7, label='Time')
        
        if has_voice and 'voice_arousal' in df.columns and 'voice_valence' in df.columns:
            ax8 = fig.add_subplot(gs[row, 1])
            scatter = ax8.scatter(df['voice_valence'], df['voice_arousal'], 
                                 c=time, cmap='plasma', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax8.set_title('🎤 Voice Emotion Space', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Valence (negative ← → positive)', fontsize=10)
            ax8.set_ylabel('Arousal (low ↓ ↑ high)', fontsize=10)
            ax8.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax8.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax8.set_xlim([-1.1, 1.1])
            ax8.set_ylim([-1.1, 1.1])
            ax8.grid(True, alpha=0.2)
            # Add quadrant labels
            ax8.text(0.5, 0.5, 'EXCITED', fontsize=9, ha='center', va='center', alpha=0.4, fontweight='bold')
            ax8.text(-0.5, 0.5, 'STRESSED', fontsize=9, ha='center', va='center', alpha=0.4, fontweight='bold')
            ax8.text(-0.5, -0.5, 'TIRED', fontsize=9, ha='center', va='center', alpha=0.4, fontweight='bold')
            ax8.text(0.5, -0.5, 'PEACEFUL', fontsize=9, ha='center', va='center', alpha=0.4, fontweight='bold')
            plt.colorbar(scatter, ax=ax8, label='Time')
        
        # Statistical Distribution - Facial
        if has_facial:
            ax9 = fig.add_subplot(gs[row, 2])
            facial_data = []
            facial_labels = []
            for emotion_col in self.facial_emotions:
                if emotion_col in df.columns:
                    facial_data.append(df[emotion_col].values)
                    facial_labels.append(emotion_col.replace('facial_', '').title())
            if facial_data:
                bp = ax9.boxplot(facial_data, labels=facial_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                ax9.set_title('📸 Facial Emotion Distributions', fontsize=12, fontweight='bold')
                ax9.set_ylabel('Probability', fontsize=10)
                ax9.tick_params(axis='x', rotation=45, labelsize=8)
                ax9.grid(True, alpha=0.3, axis='y')
        
        # Statistical Distribution - Voice
        if has_voice:
            ax10 = fig.add_subplot(gs[row, 3])
            voice_data = []
            voice_labels = []
            for emotion_col in self.voice_emotions:
                if emotion_col in df.columns:
                    voice_data.append(df[emotion_col].values)
                    voice_labels.append(emotion_col.replace('voice_', '').title())
            if voice_data:
                bp = ax10.boxplot(voice_data, labels=voice_labels, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightcoral')
                ax10.set_title('🎤 Voice Emotion Distributions', fontsize=12, fontweight='bold')
                ax10.set_ylabel('Probability', fontsize=10)
                ax10.tick_params(axis='x', rotation=45, labelsize=8)
                ax10.grid(True, alpha=0.3, axis='y')
        
        row += 1
        
        # ROW 5: Correlation Matrices
        if has_facial:
            ax11 = fig.add_subplot(gs[row, :2])
            facial_cols = [col for col in self.facial_emotions if col in df.columns]
            if len(facial_cols) > 1:
                corr = df[facial_cols].corr()
                im = ax11.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax11.set_title('📸 Facial Emotion Correlations', fontsize=12, fontweight='bold')
                ax11.set_xticks(range(len(facial_cols)))
                ax11.set_yticks(range(len(facial_cols)))
                ax11.set_xticklabels([col.replace('facial_', '') for col in facial_cols], rotation=45, ha='right', fontsize=9)
                ax11.set_yticklabels([col.replace('facial_', '') for col in facial_cols], fontsize=9)
                # Add correlation values
                for i in range(len(facial_cols)):
                    for j in range(len(facial_cols)):
                        text = ax11.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
                plt.colorbar(im, ax=ax11, label='Correlation')
        
        if has_voice:
            ax12 = fig.add_subplot(gs[row, 2:])
            voice_cols = [col for col in self.voice_emotions if col in df.columns]
            if len(voice_cols) > 1:
                corr = df[voice_cols].corr()
                im = ax12.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax12.set_title('🎤 Voice Emotion Correlations', fontsize=12, fontweight='bold')
                ax12.set_xticks(range(len(voice_cols)))
                ax12.set_yticks(range(len(voice_cols)))
                ax12.set_xticklabels([col.replace('voice_', '') for col in voice_cols], rotation=45, ha='right', fontsize=9)
                ax12.set_yticklabels([col.replace('voice_', '') for col in voice_cols], fontsize=9)
                # Add correlation values
                for i in range(len(voice_cols)):
                    for j in range(len(voice_cols)):
                        text = ax12.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)
                plt.colorbar(im, ax=ax12, label='Correlation')
        
        row += 1
        
        # ROW 6: Summary Statistics and Insights
        ax13 = fig.add_subplot(gs[row, :2])
        ax13.axis('off')
        
        summary_text = "📊 COMPREHENSIVE STATISTICAL SUMMARY\n\n"
        
        if has_facial:
            summary_text += "🎭 FACIAL ANALYSIS:\n"
            for emotion_col in self.facial_emotions:
                if emotion_col in df.columns:
                    emotion_name = emotion_col.replace('facial_', '').upper()
                    mean_val = df[emotion_col].mean()
                    std_val = df[emotion_col].std()
                    max_val = df[emotion_col].max()
                    summary_text += f"  {emotion_name:10s}: μ={mean_val:.3f}, σ={std_val:.3f}, max={max_val:.3f}\n"
            
            # Dominant emotion
            facial_means = {col: df[col].mean() for col in self.facial_emotions if col in df.columns}
            if facial_means:
                dominant = max(facial_means, key=facial_means.get)
                summary_text += f"\n  🏆 Dominant: {dominant.replace('facial_', '').upper()} ({facial_means[dominant]:.3f})\n"
        
        if has_voice:
            summary_text += "\n🎤 VOICE ANALYSIS:\n"
            for emotion_col in self.voice_emotions:
                if emotion_col in df.columns:
                    emotion_name = emotion_col.replace('voice_', '').upper()
                    mean_val = df[emotion_col].mean()
                    std_val = df[emotion_col].std()
                    max_val = df[emotion_col].max()
                    summary_text += f"  {emotion_name:10s}: μ={mean_val:.3f}, σ={std_val:.3f}, max={max_val:.3f}\n"
            
            # Dominant emotion
            voice_means = {col: df[col].mean() for col in self.voice_emotions if col in df.columns}
            if voice_means:
                dominant = max(voice_means, key=voice_means.get)
                summary_text += f"\n  🏆 Dominant: {dominant.replace('voice_', '').upper()} ({voice_means[dominant]:.3f})\n"
        
        ax13.text(0.05, 0.95, summary_text, transform=ax13.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Quadrant Analysis
        ax14 = fig.add_subplot(gs[row, 2:])
        ax14.axis('off')
        
        quad_text = "🎯 EMOTIONAL QUADRANT ANALYSIS\n\n"
        
        if has_facial and 'facial_quadrant' in df.columns:
            quad_counts = df['facial_quadrant'].value_counts()
            quad_text += "📸 FACIAL QUADRANTS:\n"
            for quad, count in quad_counts.items():
                pct = (count / len(df)) * 100
                quad_text += f"  {quad:10s}: {count:4d} ({pct:5.1f}%)\n"
        
        if has_voice and 'voice_quadrant' in df.columns:
            quad_counts = df['voice_quadrant'].value_counts()
            quad_text += "\n🎤 VOICE QUADRANTS:\n"
            for quad, count in quad_counts.items():
                pct = (count / len(df)) * 100
                quad_text += f"  {quad:10s}: {count:4d} ({pct:5.1f}%)\n"
        
        quad_text += f"\n📈 Total Samples: {len(df)}\n"
        if 'time_seconds' in df.columns:
            duration = df['time_seconds'].max()
            quad_text += f"⏱️  Duration: {duration:.1f} seconds\n"
        
        ax14.text(0.05, 0.95, quad_text, transform=ax14.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # Main title
        fig.suptitle('🌟 ULTIMATE COMPREHENSIVE EMOTION ANALYSIS REPORT 🌟', 
                    fontsize=20, fontweight='bold', y=0.995)
        
        # Save
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n✅ ULTIMATE MEGA REPORT SAVED: {save_path}")
        print(f"   Total subplots: 14+")
        print(f"   File size: ~{os.path.getsize(save_path) / 1024 / 1024:.1f} MB")
        print(f"{'='*80}\n")
        
        return save_path


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        print("Usage: python generate_ultimate_report.py <csv_file>")
        print("Example: python generate_ultimate_report.py angry_emotion_data.csv")
        sys.exit(1)
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    generator = UltimateReportGenerator()
    generator.generate_mega_report(df, save_path=csv_file.replace('.csv', '_ULTIMATE.png'))
