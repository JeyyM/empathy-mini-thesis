"""
Comprehensive Voice Emotion Report Generator
Visualizes ALL voice/audio emotion data collected
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

class ComprehensiveVoiceReport:
    """Generate detailed reports showing ALL voice emotion data"""
    
    def __init__(self):
        self.voice_emotions = ['angry', 'happy', 'sad', 'neutral']
        self.acoustic_features = ['pitch_mean', 'pitch_std', 'volume_mean', 'volume_std', 
                                 'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate']
        self.mfcc_features = [f'mfcc_{i}' for i in range(1, 14)]  # MFCCs 1-13
        
    def generate_report(self, df, save_path="voice_comprehensive_report.png"):
        """
        Generate comprehensive voice emotion report with ALL data visualized
        
        Args:
            df: DataFrame with voice emotion data
            save_path: Path to save the report image
        """
        # Detect column prefixes (voice_ or no prefix)
        has_prefix = 'voice_happy' in df.columns
        prefix = 'voice_' if has_prefix else ''
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.35)
        
        fig.suptitle('🎤 COMPREHENSIVE VOICE EMOTION ANALYSIS - ALL DATA', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ ROW 1: Voice Emotions ============
        
        # Plot 1: All voice emotions over time
        ax1 = fig.add_subplot(gs[0, :2])
        colors = ['#FF4444', '#44FF44', '#4444FF', '#888888']
        for emotion, color in zip(self.voice_emotions, colors):
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                ax1.plot(time_data, df[col_name], label=emotion.capitalize(), 
                        color=color, linewidth=2.5, alpha=0.8)
        ax1.set_title('🗣️ Voice Emotions Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Emotion Probability (0-1)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # Plot 2: Voice arousal and valence
        ax2 = fig.add_subplot(gs[0, 2:])
        arousal_col = f'{prefix}arousal'
        valence_col = f'{prefix}valence'
        if arousal_col in df.columns and valence_col in df.columns:
            ax2.plot(time_data, df[arousal_col], label='Voice Arousal (Energy)', 
                    color='#FF6B6B', linewidth=2.5)
            ax2.plot(time_data, df[valence_col], label='Voice Valence (Mood)', 
                    color='#4ECDC4', linewidth=2.5)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title('⚡ Voice Arousal & Valence', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Level (-1 to 1)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-1.1, 1.1)
        
        # ============ ROW 2: Pitch Analysis ============
        
        # Plot 3: Pitch mean over time
        ax3 = fig.add_subplot(gs[1, 0])
        pitch_mean_col = f'{prefix}pitch_mean'
        if pitch_mean_col in df.columns:
            ax3.plot(time_data, df[pitch_mean_col], color='#9C27B0', linewidth=2)
            ax3.fill_between(time_data, df[pitch_mean_col], alpha=0.3, color='#9C27B0')
            ax3.set_title('🎵 Pitch Mean (Hz)', fontsize=13, fontweight='bold')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Frequency (Hz)')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Pitch variation (std)
        ax4 = fig.add_subplot(gs[1, 1])
        pitch_std_col = f'{prefix}pitch_std'
        if pitch_std_col in df.columns:
            ax4.plot(time_data, df[pitch_std_col], color='#673AB7', linewidth=2)
            ax4.fill_between(time_data, df[pitch_std_col], alpha=0.3, color='#673AB7')
            ax4.set_title('📊 Pitch Variation (Std Dev)', fontsize=13, fontweight='bold')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Variation (Hz)')
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Volume mean
        ax5 = fig.add_subplot(gs[1, 2])
        volume_mean_col = f'{prefix}volume_mean'
        if volume_mean_col in df.columns:
            ax5.plot(time_data, df[volume_mean_col], color='#FF5722', linewidth=2)
            ax5.fill_between(time_data, df[volume_mean_col], alpha=0.3, color='#FF5722')
            ax5.set_title('🔊 Volume Mean (dB)', fontsize=13, fontweight='bold')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('Volume (dB)')
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Volume variation
        ax6 = fig.add_subplot(gs[1, 3])
        volume_std_col = f'{prefix}volume_std'
        if volume_std_col in df.columns:
            ax6.plot(time_data, df[volume_std_col], color='#E64A19', linewidth=2)
            ax6.fill_between(time_data, df[volume_std_col], alpha=0.3, color='#E64A19')
            ax6.set_title('📈 Volume Variation (Std Dev)', fontsize=13, fontweight='bold')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Variation (dB)')
            ax6.grid(True, alpha=0.3)
        
        # ============ ROW 3: Spectral Features ============
        
        # Plot 7: Spectral Centroid
        ax7 = fig.add_subplot(gs[2, 0])
        spectral_centroid_col = f'{prefix}spectral_centroid'
        if spectral_centroid_col in df.columns:
            ax7.plot(time_data, df[spectral_centroid_col], color='#00BCD4', linewidth=2)
            ax7.set_title('✨ Spectral Centroid', fontsize=13, fontweight='bold')
            ax7.set_xlabel('Time (s)')
            ax7.set_ylabel('Frequency (Hz)')
            ax7.grid(True, alpha=0.3)
        
        # Plot 8: Spectral Rolloff
        ax8 = fig.add_subplot(gs[2, 1])
        spectral_rolloff_col = f'{prefix}spectral_rolloff'
        if spectral_rolloff_col in df.columns:
            ax8.plot(time_data, df[spectral_rolloff_col], color='#0097A7', linewidth=2)
            ax8.set_title('🌊 Spectral Rolloff', fontsize=13, fontweight='bold')
            ax8.set_xlabel('Time (s)')
            ax8.set_ylabel('Frequency (Hz)')
            ax8.grid(True, alpha=0.3)
        
        # Plot 9: Zero Crossing Rate
        ax9 = fig.add_subplot(gs[2, 2])
        zcr_col = f'{prefix}zero_crossing_rate'
        if zcr_col in df.columns:
            ax9.plot(time_data, df[zcr_col], color='#009688', linewidth=2)
            ax9.set_title('⚡ Zero Crossing Rate', fontsize=13, fontweight='bold')
            ax9.set_xlabel('Time (s)')
            ax9.set_ylabel('Rate')
            ax9.grid(True, alpha=0.3)
        
        # Plot 10: Speaking Rate (if available)
        ax10 = fig.add_subplot(gs[2, 3])
        speaking_rate_col = f'{prefix}speaking_rate'
        tempo_col = f'{prefix}tempo'
        
        if speaking_rate_col in df.columns:
            ax10.plot(time_data, df[speaking_rate_col], color='#4CAF50', linewidth=2)
            ax10.set_title('⏱️ Speaking Rate', fontsize=13, fontweight='bold')
            ax10.set_xlabel('Time (s)')
            ax10.set_ylabel('Rate')
            ax10.grid(True, alpha=0.3)
        elif tempo_col in df.columns:
            ax10.plot(time_data, df[tempo_col], color='#4CAF50', linewidth=2)
            ax10.set_title('🎼 Tempo (BPM)', fontsize=13, fontweight='bold')
            ax10.set_xlabel('Time (s)')
            ax10.set_ylabel('BPM')
            ax10.grid(True, alpha=0.3)
        else:
            ax10.text(0.5, 0.5, 'No tempo/rate data', ha='center', va='center', fontsize=12)
            ax10.axis('off')
        
        # ============ ROW 4: MFCC Analysis ============
        
        # Plot 11: MFCC heatmap (first 13 coefficients)
        ax11 = fig.add_subplot(gs[3, :3])
        mfcc_data = []
        available_mfccs = []
        
        for i in range(1, 14):
            mfcc_col = f'{prefix}mfcc_{i}'
            if mfcc_col in df.columns:
                mfcc_data.append(df[mfcc_col].values)
                available_mfccs.append(f'MFCC {i}')
        
        if mfcc_data:
            im = ax11.imshow(mfcc_data, aspect='auto', cmap='viridis', 
                           interpolation='nearest', 
                           extent=[time_data.min(), time_data.max(), 0, len(available_mfccs)])
            ax11.set_yticks(np.arange(len(available_mfccs)) + 0.5)
            ax11.set_yticklabels(available_mfccs)
            ax11.set_xlabel('Time (seconds)')
            ax11.set_title('🎨 MFCC Coefficients Heatmap (Audio Fingerprint)', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax11, label='Coefficient Value')
        
        # Plot 12: Voice emotion distribution
        ax12 = fig.add_subplot(gs[3, 3])
        emotion_avgs = {}
        for emotion in self.voice_emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                emotion_avgs[emotion] = df[col_name].mean()
        
        if emotion_avgs:
            colors_pie = ['#FF4444', '#44FF44', '#4444FF', '#888888']
            ax12.pie(emotion_avgs.values(), labels=[e.capitalize() for e in emotion_avgs.keys()],
                    autopct='%1.1f%%', colors=colors_pie[:len(emotion_avgs)], startangle=90)
            ax12.set_title('🥧 Voice Emotion Distribution', fontsize=13, fontweight='bold')
        
        # ============ ROW 5: Statistical Summary ============
        
        # Plot 13: Feature statistics table
        ax13 = fig.add_subplot(gs[4, :2])
        ax13.axis('off')
        
        # Acoustic features stats
        stats_data = []
        feature_names = ['Pitch Mean', 'Pitch Std', 'Volume Mean', 'Volume Std', 
                        'Spectral Centroid', 'Spectral Rolloff', 'Zero Cross Rate']
        
        for feature_name, col_suffix in zip(feature_names, 
                                            ['pitch_mean', 'pitch_std', 'volume_mean', 'volume_std',
                                             'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate']):
            col_name = f'{prefix}{col_suffix}'
            if col_name in df.columns:
                stats_data.append([
                    feature_name,
                    f"{df[col_name].mean():.2f}",
                    f"{df[col_name].std():.2f}",
                    f"{df[col_name].min():.2f}",
                    f"{df[col_name].max():.2f}"
                ])
        
        if stats_data:
            table = ax13.table(cellText=stats_data,
                            colLabels=['Feature', 'Mean', 'Std Dev', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.5)
            
            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#00BCD4')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(stats_data) + 1):
                for j in range(5):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#F0F0F0')
        
        ax13.set_title('📊 Acoustic Features - Statistical Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Plot 14: Prosody features (if available)
        ax14 = fig.add_subplot(gs[4, 2:])
        
        # Check for prosody features
        prosody_features = []
        prosody_values = []
        prosody_labels = []
        
        prosody_cols = ['pitch_range', 'energy', 'intensity', 'jitter', 'shimmer']
        prosody_names = ['Pitch Range', 'Energy', 'Intensity', 'Jitter', 'Shimmer']
        
        for col_suffix, label in zip(prosody_cols, prosody_names):
            col_name = f'{prefix}{col_suffix}'
            if col_name in df.columns:
                prosody_features.append(df[col_name].mean())
                prosody_labels.append(label)
        
        if prosody_features:
            bars = ax14.barh(range(len(prosody_labels)), prosody_features, color='#FF9800')
            ax14.set_yticks(range(len(prosody_labels)))
            ax14.set_yticklabels(prosody_labels)
            ax14.set_xlabel('Average Value')
            ax14.set_title('🎭 Prosody Features (Average)', fontsize=14, fontweight='bold')
            ax14.grid(True, alpha=0.3, axis='x')
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars, prosody_features)):
                ax14.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=10)
        else:
            ax14.text(0.5, 0.5, 'Voice Emotion Distribution Summary', 
                     ha='center', va='top', fontsize=12, fontweight='bold')
            
            # Show emotion statistics instead
            y_pos = 0.4
            for emotion in self.voice_emotions:
                col_name = f'{prefix}{emotion}'
                if col_name in df.columns:
                    avg = df[col_name].mean()
                    ax14.text(0.5, y_pos, f'{emotion.capitalize()}: {avg:.3f}', 
                            ha='center', va='center', fontsize=11)
                    y_pos -= 0.1
            ax14.axis('off')
        
        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Comprehensive voice report saved to: {save_path}")
        plt.close()
        
        # Print summary statistics
        self.print_summary(df, prefix)
    
    def print_summary(self, df, prefix=''):
        """Print detailed summary statistics"""
        print(f"\n{'='*80}")
        print("🎤 VOICE EMOTION ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"Total samples: {len(df)}")
        
        if 'time_seconds' in df.columns:
            duration = df['time_seconds'].max() - df['time_seconds'].min()
            print(f"Duration: {duration:.1f} seconds")
        
        print(f"\n--- Voice Emotion Averages ---")
        for emotion in self.voice_emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                avg = df[col_name].mean()
                print(f"{emotion.capitalize():>8s}: {avg:.3f}")
        
        print(f"\n--- Acoustic Feature Averages ---")
        pitch_mean_col = f'{prefix}pitch_mean'
        volume_mean_col = f'{prefix}volume_mean'
        spectral_centroid_col = f'{prefix}spectral_centroid'
        
        if pitch_mean_col in df.columns:
            print(f"Pitch Mean: {df[pitch_mean_col].mean():.2f} Hz")
        if volume_mean_col in df.columns:
            print(f"Volume Mean: {df[volume_mean_col].mean():.2f} dB")
        if spectral_centroid_col in df.columns:
            print(f"Spectral Centroid: {df[spectral_centroid_col].mean():.2f} Hz")
        
        print(f"\n--- Dominant Voice Emotion ---")
        emotion_avgs = {}
        for emotion in self.voice_emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                emotion_avgs[emotion] = df[col_name].mean()
        
        if emotion_avgs:
            dominant = max(emotion_avgs.items(), key=lambda x: x[1])
            print(f"{dominant[0].capitalize()}: {dominant[1]:.3f}")
        
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
        reporter = ComprehensiveVoiceReport()
        reporter.generate_report(df, save_path="voice_comprehensive_report.png")
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("Usage: python generate_comprehensive_voice_report.py [csv_file]")
