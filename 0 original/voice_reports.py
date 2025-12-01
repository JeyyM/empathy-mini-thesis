"""
Modular Voice Emotion Report Generator
Generates separate detailed reports for different aspects of voice emotion analysis
Color-coded according to emotion/dimension specifications
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import seaborn as sns

class VoiceReportGenerator:
    """Generate modular voice emotion reports"""
    
    def __init__(self):
        # Color coding for emotions
        self.emotion_colors = {
            'angry': '#E74C3C',      # Red
            'surprise': '#FF8C00',   # Orange
            'happy': '#FFD700',      # Yellow
            'disgust': '#2ECC71',    # Green
            'sad': '#3498DB',        # Blue
            'fear': '#9370DB',       # Violet
            'neutral': '#95A5A6'     # Ash Gray
        }
        
        # Color coding for dimensions
        self.dimension_colors = {
            'intensity': '#E74C3C',  # Red
            'valence': '#3498DB',    # Blue
            'arousal': '#2ECC71',    # Green
            'stress': '#FF8C00'      # Orange (for stress)
        }
        
        # Color coding for emotional states
        self.state_colors = {
            'Stressed': '#FF8C00',   # Orange
            'Excited': '#FFD700',    # Yellow
            'Tired': '#3498DB',      # Blue
            'Calm': '#2ECC71'        # Green
        }
        
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def generate_emotion_report(self, df, base_filename):
        """
        Generate Page 1: Voice Emotion Analysis Report
        - Emotion over time of all 7 emotions
        - Highest overall emotion line graph
        - Summary statistics of all 7 emotions
        - Proportion pie graph and values
        """
        # Detect column prefix
        has_prefix = 'voice_angry' in df.columns
        prefix = 'voice_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        fig.suptitle('🎤 VOICE EMOTION ANALYSIS - EMOTIONS', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ PLOT 1: All 7 Emotions Over Time ============
        ax1 = fig.add_subplot(gs[0, :])
        
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                ax1.plot(time_data, df[col_name], 
                        label=emotion.capitalize(), 
                        color=self.emotion_colors[emotion],
                        linewidth=2.5, alpha=0.8)
        
        ax1.set_title('📊 All 7 Voice Emotions Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Emotion Probability (0-1)', fontsize=12)
        ax1.legend(loc='upper right', ncol=4, fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # ============ PLOT 2: Dominant Emotion Only ============
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Find dominant emotion at each time point
        emotion_cols = [f'{prefix}{e}' for e in self.emotions if f'{prefix}{e}' in df.columns]
        if emotion_cols:
            dominant_values = df[emotion_cols].max(axis=1)
            dominant_emotions = df[emotion_cols].idxmax(axis=1)
            
            # Plot with color coding
            for i in range(len(time_data)):
                emotion_name = dominant_emotions.iloc[i].replace(prefix, '')
                color = self.emotion_colors.get(emotion_name, '#95A5A6')
                if i < len(time_data) - 1:
                    ax2.plot(time_data.iloc[i:i+2], dominant_values.iloc[i:i+2], 
                            color=color, linewidth=3, alpha=0.9)
        
        ax2.set_title('⭐ Dominant Voice Emotion Over Time (Color-Coded)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Emotion Strength (0-1)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.emotion_colors[e], label=e.capitalize()) 
                          for e in self.emotions]
        ax2.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=9)
        
        # ============ PLOT 3: Summary Statistics Table ============
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        stats_data = []
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                stats_data.append([
                    emotion.capitalize(),
                    f"{df[col_name].mean():.3f}",
                    f"{df[col_name].std():.3f}",
                    f"{df[col_name].median():.3f}",
                    f"{df[col_name].min():.3f}",
                    f"{df[col_name].max():.3f}"
                ])
        
        if stats_data:
            table = ax3.table(cellText=stats_data,
                            colLabels=['Emotion', 'Mean', 'Std Dev', 'Median', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Color code rows by emotion
            for i, emotion in enumerate([e for e in self.emotions if f'{prefix}{e}' in df.columns], 1):
                table[(i, 0)].set_facecolor(self.emotion_colors[emotion])
                table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(6):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('📈 Summary Statistics (All 7 Voice Emotions)', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 4: Proportion Pie Chart ============
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Calculate average proportions
        emotion_avgs = {}
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                emotion_avgs[emotion] = df[col_name].mean()
        
        if emotion_avgs:
            colors = [self.emotion_colors[e] for e in emotion_avgs.keys()]
            labels = [f"{e.capitalize()}\n{v:.3f} ({v*100:.1f}%)" 
                     for e, v in emotion_avgs.items()]
            
            ax4.pie(emotion_avgs.values(), 
                   labels=labels,
                   colors=colors,
                   autopct='',
                   startangle=90,
                   textprops={'fontsize': 11, 'weight': 'bold'})
            
            ax4.set_title('🥧 Voice Emotion Distribution', 
                         fontsize=16, fontweight='bold', pad=20)
        
        # ============ PLOT 5: Distribution Table ============
        ax5 = fig.add_subplot(gs[2, 1:])
        ax5.axis('off')
        
        if emotion_avgs:
            # Create table data
            table_data = []
            total_avg = sum(emotion_avgs.values())
            
            for emotion in self.emotions:
                if emotion in emotion_avgs:
                    avg_value = emotion_avgs[emotion]
                    proportion = avg_value / total_avg if total_avg > 0 else 0
                    percentage = proportion * 100
                    table_data.append([
                        emotion.capitalize(),
                        f'{proportion:.4f}',
                        f'{percentage:.2f}%'
                    ])
            
            table = ax5.table(cellText=table_data,
                            colLabels=['Emotion', 'Proportion', 'Percentage'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Color code emotion names
            for i, emotion in enumerate([e for e in self.emotions if e in emotion_avgs], 1):
                table[(i, 0)].set_facecolor(self.emotion_colors[emotion])
                table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(3):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            ax5.set_title('📊 Emotion Distribution Table', fontsize=14, fontweight='bold', pad=20)
            
            # Add dominant emotion text at bottom
            dominant_emotion = max(emotion_avgs.items(), key=lambda x: x[1])
            fig.text(0.5, 0.02, 
                    f"Most Dominant: {dominant_emotion[0].capitalize()} ({dominant_emotion[1]:.4f} - {dominant_emotion[1]*100:.2f}%)",
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor=self.emotion_colors[dominant_emotion[0]], 
                             edgecolor='black', linewidth=2))
        
        # Save
        save_path = f"{base_filename}_voice_emotions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Voice Emotions report saved: {save_path}")
        plt.close()
    
    def generate_acoustic_report(self, df, base_filename):
        """
        Generate Page 2: Acoustic Features Report
        - Arousal, Stress, Valence, Intensity over time
        - Averages for all four
        - Pitch mean and variation
        - Volume mean and variation
        - Spectral centroid and rolloff
        - Zero crossing rate
        - Acoustic features summary statistics
        - MFCC coefficients
        """
        # Detect column prefix
        has_prefix = 'voice_arousal' in df.columns
        prefix = 'voice_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.35)
        
        fig.suptitle('🎤 VOICE EMOTION ANALYSIS - ACOUSTIC FEATURES', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ PLOT 1: Arousal, Stress, Valence, Intensity Over Time ============
        ax1 = fig.add_subplot(gs[0, :])
        
        dimensions = ['arousal', 'stress', 'valence', 'intensity']
        dim_labels = ['Arousal', 'Stress', 'Valence', 'Intensity']
        for dim, label in zip(dimensions, dim_labels):
            col_name = f'{prefix}{dim}'
            if col_name in df.columns:
                color = self.dimension_colors.get(dim, '#95A5A6')
                ax1.plot(time_data, df[col_name], 
                        label=label,
                        color=color,
                        linewidth=3, alpha=0.8)
        
        ax1.set_title('📊 Voice Dimensions Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Dimension Value', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # ============ PLOT 2: Dimension Averages ============
        ax2 = fig.add_subplot(gs[1, 0])
        
        dim_avgs = {}
        for dim in dimensions:
            col_name = f'{prefix}{dim}'
            if col_name in df.columns:
                dim_avgs[dim] = df[col_name].mean()
        
        if dim_avgs:
            colors = [self.dimension_colors.get(d, '#95A5A6') for d in dim_avgs.keys()]
            bars = ax2.bar(range(len(dim_avgs)), dim_avgs.values(),
                          color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax2.set_xticks(range(len(dim_avgs)))
            ax2.set_xticklabels([d.capitalize() for d in dim_avgs.keys()], 
                              fontsize=11, fontweight='bold', rotation=15)
            ax2.set_ylabel('Average Value', fontsize=12)
            ax2.set_title('📈 Dimension Averages', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Add value labels
            for i, (dim, val) in enumerate(dim_avgs.items()):
                ax2.text(i, val + 0.05 if val >= 0 else val - 0.05, 
                        f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        # ============ PLOT 3: Pitch Mean ============
        ax3 = fig.add_subplot(gs[1, 1])
        
        if f'{prefix}pitch_mean' in df.columns:
            ax3.plot(time_data, df[f'{prefix}pitch_mean'], 
                    color='#9370DB', linewidth=2.5, alpha=0.8)
            ax3.fill_between(time_data, df[f'{prefix}pitch_mean'], 0,
                           color='#9370DB', alpha=0.2)
            ax3.set_title('🎵 Pitch Mean (Hz)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Time (seconds)', fontsize=11)
            ax3.set_ylabel('Frequency (Hz)', fontsize=11)
            ax3.grid(True, alpha=0.3)
            
            # Add average line
            avg_pitch = df[f'{prefix}pitch_mean'].mean()
            ax3.axhline(y=avg_pitch, color='red', linestyle='--', linewidth=2, 
                       label=f'Avg: {avg_pitch:.1f} Hz')
            ax3.legend(fontsize=10)
        
        # ============ PLOT 4: Pitch Variation ============
        ax4 = fig.add_subplot(gs[1, 2])
        
        if f'{prefix}pitch_std' in df.columns:
            ax4.plot(time_data, df[f'{prefix}pitch_std'], 
                    color='#8B4789', linewidth=2.5, alpha=0.8)
            ax4.fill_between(time_data, df[f'{prefix}pitch_std'], 0,
                           color='#8B4789', alpha=0.2)
            ax4.set_title('📊 Pitch Variation (Std Dev)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Time (seconds)', fontsize=11)
            ax4.set_ylabel('Standard Deviation (Hz)', fontsize=11)
            ax4.grid(True, alpha=0.3)
            
            # Add average line
            avg_std = df[f'{prefix}pitch_std'].mean()
            ax4.axhline(y=avg_std, color='red', linestyle='--', linewidth=2,
                       label=f'Avg: {avg_std:.1f} Hz')
            ax4.legend(fontsize=10)
        
        # ============ PLOT 5: Volume Mean ============
        ax5 = fig.add_subplot(gs[2, 0])
        
        if f'{prefix}volume_mean' in df.columns:
            ax5.plot(time_data, df[f'{prefix}volume_mean'], 
                    color='#FF8C00', linewidth=2.5, alpha=0.8)
            ax5.fill_between(time_data, df[f'{prefix}volume_mean'], 0,
                           color='#FF8C00', alpha=0.2)
            ax5.set_title('🔊 Volume Mean (dB)', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Time (seconds)', fontsize=11)
            ax5.set_ylabel('Amplitude (dB)', fontsize=11)
            ax5.grid(True, alpha=0.3)
            
            avg_vol = df[f'{prefix}volume_mean'].mean()
            ax5.axhline(y=avg_vol, color='red', linestyle='--', linewidth=2,
                       label=f'Avg: {avg_vol:.3f} dB')
            ax5.legend(fontsize=10)
        
        # ============ PLOT 6: Volume Variation ============
        ax6 = fig.add_subplot(gs[2, 1])
        
        if f'{prefix}volume_std' in df.columns:
            ax6.plot(time_data, df[f'{prefix}volume_std'], 
                    color='#E64A19', linewidth=2.5, alpha=0.8)
            ax6.fill_between(time_data, df[f'{prefix}volume_std'], 0,
                           color='#E64A19', alpha=0.2)
            ax6.set_title('📊 Volume Variation (Std Dev)', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Time (seconds)', fontsize=11)
            ax6.set_ylabel('Standard Deviation (dB)', fontsize=11)
            ax6.grid(True, alpha=0.3)
            
            avg_vol_std = df[f'{prefix}volume_std'].mean()
            ax6.axhline(y=avg_vol_std, color='red', linestyle='--', linewidth=2,
                       label=f'Avg: {avg_vol_std:.3f} dB')
            ax6.legend(fontsize=10)
        
        # ============ PLOT 7: Spectral Features ============
        ax7 = fig.add_subplot(gs[2, 2])
        
        spectral_features = []
        if f'{prefix}spectral_centroid' in df.columns:
            ax7.plot(time_data, df[f'{prefix}spectral_centroid'], 
                    label='Spectral Centroid', color='#00BCD4', linewidth=2.5, alpha=0.8)
            spectral_features.append('centroid')
        
        if f'{prefix}spectral_rolloff' in df.columns:
            ax7.plot(time_data, df[f'{prefix}spectral_rolloff'], 
                    label='Spectral Rolloff', color='#0097A7', linewidth=2.5, alpha=0.8)
            spectral_features.append('rolloff')
        
        if spectral_features:
            ax7.set_title('✨ Spectral Features (Hz)', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Time (seconds)', fontsize=11)
            ax7.set_ylabel('Frequency (Hz)', fontsize=11)
            ax7.legend(fontsize=10)
            ax7.grid(True, alpha=0.3)
        
        # ============ PLOT 8: Zero Crossing Rate ============
        ax8 = fig.add_subplot(gs[3, 0])
        
        if f'{prefix}zero_crossing_rate' in df.columns:
            ax8.plot(time_data, df[f'{prefix}zero_crossing_rate'], 
                    color='#009688', linewidth=2.5, alpha=0.8)
            ax8.fill_between(time_data, df[f'{prefix}zero_crossing_rate'], 0,
                           color='#009688', alpha=0.2)
            ax8.set_title('🌊 Zero Crossing Rate', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Time (seconds)', fontsize=11)
            ax8.set_ylabel('Rate', fontsize=11)
            ax8.grid(True, alpha=0.3)
            
            avg_zcr = df[f'{prefix}zero_crossing_rate'].mean()
            ax8.axhline(y=avg_zcr, color='red', linestyle='--', linewidth=2,
                       label=f'Avg: {avg_zcr:.4f}')
            ax8.legend(fontsize=10)
        
        # ============ PLOT 9: Acoustic Features Summary Statistics ============
        ax9 = fig.add_subplot(gs[3, 1:])
        ax9.axis('off')
        
        acoustic_features = {
            'Pitch Mean (Hz)': f'{prefix}pitch_mean',
            'Pitch Std Dev (Hz)': f'{prefix}pitch_std',
            'Volume Mean (dB)': f'{prefix}volume_mean',
            'Volume Std Dev (dB)': f'{prefix}volume_std',
            'Spectral Centroid (Hz)': f'{prefix}spectral_centroid',
            'Spectral Rolloff (Hz)': f'{prefix}spectral_rolloff',
            'Zero Crossing Rate': f'{prefix}zero_crossing_rate'
        }
        
        stats_data = []
        for feature_name, col_name in acoustic_features.items():
            if col_name in df.columns:
                stats_data.append([
                    feature_name,
                    f"{df[col_name].mean():.3f}",
                    f"{df[col_name].std():.3f}",
                    f"{df[col_name].median():.3f}",
                    f"{df[col_name].min():.3f}",
                    f"{df[col_name].max():.3f}",
                    f"{df[col_name].max() - df[col_name].min():.3f}"
                ])
        
        if stats_data:
            table = ax9.table(cellText=stats_data,
                            colLabels=['Feature', 'Mean', 'Std Dev', 'Median', 'Min', 'Max', 'Range'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.5)
            
            # Style header
            for j in range(7):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(stats_data) + 1):
                for j in range(7):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#ECF0F1')
        
        ax9.set_title('📊 Acoustic Features - Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 10: MFCC Coefficients Heatmap ============
        ax10 = fig.add_subplot(gs[4, :])
        
        # Find MFCC columns (exclude _std, _mean, _max, _min)
        mfcc_cols = [col for col in df.columns if 'mfcc_' in col and not any(x in col for x in ['_std', '_mean', '_max', '_min'])]
        
        if mfcc_cols:
            # Sort by MFCC number
            mfcc_cols_sorted = sorted(mfcc_cols, key=lambda x: int(x.split('_')[-1]))
            
            # Create heatmap data
            mfcc_data = df[mfcc_cols_sorted].T.values
            
            im = ax10.imshow(mfcc_data, aspect='auto', cmap='viridis', interpolation='nearest')
            ax10.set_xlabel('Time (samples)', fontsize=12)
            ax10.set_ylabel('MFCC Coefficient', fontsize=12)
            ax10.set_title('🎨 MFCC Coefficients Heatmap (Audio Fingerprint)', 
                          fontsize=16, fontweight='bold')
            
            # Set y-axis labels
            ax10.set_yticks(range(len(mfcc_cols_sorted)))
            ax10.set_yticklabels([col.replace(prefix, '').replace('_', ' ').upper() 
                                 for col in mfcc_cols_sorted], fontsize=9)
            
            plt.colorbar(im, ax=ax10, label='Coefficient Value')
        
        # Save
        save_path = f"{base_filename}_voice_acoustic.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Voice Acoustic Features report saved: {save_path}")
        plt.close()
    
    def generate_dimensions_report(self, df, base_filename):
        """
        Generate Page 3: Dimensions Report
        - Arousal, Valence, Intensity, Stress over time
        - Dimension averages and statistics
        - Emotion stability metrics
        - Emotional volatility over time
        """
        # Detect column prefix
        has_prefix = 'voice_arousal' in df.columns
        prefix = 'voice_' if has_prefix else ''
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        fig.suptitle('🎤 VOICE EMOTION ANALYSIS - DIMENSIONS', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # ============ PLOT 1: Arousal, Valence, Intensity, Stress Over Time ============
        ax1 = fig.add_subplot(gs[0, :])
        
        dimensions = ['arousal', 'valence', 'intensity', 'stress']
        for dim in dimensions:
            col_name = f'{prefix}{dim}'
            if col_name in df.columns:
                color = self.dimension_colors.get(dim, '#95A5A6')
                ax1.plot(time_data, df[col_name], 
                        label=dim.capitalize(),
                        color=color,
                        linewidth=3, alpha=0.8)
        
        ax1.set_title('📊 Voice Dimensions Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # ============ PLOT 2: Dimension Statistics Table ============
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        stats_data = []
        for dim in dimensions:
            col_name = f'{prefix}{dim}'
            if col_name in df.columns:
                stats_data.append([
                    dim.capitalize(),
                    f"{df[col_name].mean():.4f}",
                    f"{df[col_name].std():.4f}",
                    f"{df[col_name].min():.4f}",
                    f"{df[col_name].max():.4f}"
                ])
        
        if stats_data:
            table = ax2.table(cellText=stats_data,
                            colLabels=['Dimension', 'Mean', 'Std Dev', 'Min', 'Max'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 3)
            
            # Color code dimension names
            for i, dim in enumerate(dimensions, 1):
                if i <= len(stats_data):
                    table[(i, 0)].set_facecolor(self.dimension_colors.get(dim, '#95A5A6'))
                    table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(5):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('📊 Dimension Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        ax2.set_title('📊 Dimension Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 3: Emotion Stability Metrics ============
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.axis('off')
        
        # Calculate stability metrics
        stability_data = []
        
        # Count emotion changes
        emotion_cols = [f'{prefix}{e}' for e in self.emotions if f'{prefix}{e}' in df.columns]
        if emotion_cols:
            dominant_emotions = df[emotion_cols].idxmax(axis=1)
            emotion_changes = (dominant_emotions != dominant_emotions.shift()).sum() - 1
            stability_data.append(['Emotion Changes', f'{emotion_changes}'])
        
        # Standard deviation for each emotion
        for emotion in self.emotions:
            col_name = f'{prefix}{emotion}'
            if col_name in df.columns:
                stability_data.append([f'{emotion.capitalize()} Std Dev', f'{df[col_name].std():.4f}'])
        
        # Overall volatility
        if emotion_cols:
            volatility = df[emotion_cols].std().mean()
            stability_data.append(['Overall Volatility', f'{volatility:.4f}'])
        
        if stability_data:
            table = ax3.table(cellText=stability_data,
                            colLabels=['Stability Metric', 'Value'],
                            cellLoc='left',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.5)
            
            # Style header
            for j in range(2):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(stability_data) + 1):
                for j in range(2):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#ECF0F1')
        
        ax3.set_title('📊 Emotion Stability Metrics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 4: Emotional Volatility Over Time ============
        ax4 = fig.add_subplot(gs[2, 1])
        
        if emotion_cols:
            # Calculate rolling standard deviation
            window = min(30, len(df) // 10)
            rolling_std = df[emotion_cols].rolling(window=window, center=True).std().mean(axis=1)
            
            ax4.plot(time_data, rolling_std, color='#E74C3C', linewidth=2.5, alpha=0.8)
            ax4.fill_between(time_data, rolling_std, 0, color='#E74C3C', alpha=0.2)
            ax4.set_title(f'📈 Voice Emotional Volatility Over Time (Rolling Std Dev, window={window})', 
                         fontsize=14, fontweight='bold')
            ax4.set_xlabel('Time (seconds)', fontsize=12)
            ax4.set_ylabel('Standard Deviation', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        # Save
        save_path = f"{base_filename}_voice_dimensions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Voice Dimensions report saved: {save_path}")
        plt.close()
    
    def generate_states_report(self, df, base_filename):
        """
        Generate Page 4: Emotional States Report
        - Emotional journey path (circular)
        - State summary statistics
        - Dimension statistics
        - Average state values
        - State over time
        - State proportion pie chart
        - State distribution table
        """
        # Detect column prefix
        has_prefix = 'voice_arousal' in df.columns
        prefix = 'voice_' if has_prefix else ''
        
        # Create figure with 4x4 grid
        fig = plt.figure(figsize=(24, 20))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)
        
        fig.suptitle('🎤 VOICE EMOTION ANALYSIS - EMOTIONAL STATES', 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Get time data
        if 'time_seconds' in df.columns:
            time_data = df['time_seconds']
        else:
            time_data = np.arange(len(df))
        
        # Check if quadrant data exists, if not create it
        quadrant_col = f'{prefix}quadrant'
        if quadrant_col not in df.columns:
            # Create quadrants from arousal and valence
            if f'{prefix}arousal' in df.columns and f'{prefix}valence' in df.columns:
                df[quadrant_col] = df.apply(lambda row: 
                    'Excited' if row[f'{prefix}arousal'] > 0 and row[f'{prefix}valence'] > 0
                    else 'Stressed' if row[f'{prefix}arousal'] > 0 and row[f'{prefix}valence'] < 0
                    else 'Calm' if row[f'{prefix}arousal'] < 0 and row[f'{prefix}valence'] > 0
                    else 'Tired', axis=1)
        
        # ============ PLOT 1: Emotional Journey Path (Circular) ============
        ax1 = fig.add_subplot(gs[0:2, 0:2])  # Takes 2 rows, 2 columns
        
        if f'{prefix}arousal' in df.columns and f'{prefix}valence' in df.columns:
            arousal = df[f'{prefix}arousal'].values
            valence = df[f'{prefix}valence'].values
            
            # Create circular grid for background heatmap
            grid_size = 50
            x = np.linspace(-1, 1, grid_size)
            y = np.linspace(-1, 1, grid_size)
            X, Y = np.meshgrid(x, y)
            
            # Create mask for circular boundary
            circle_mask = X**2 + Y**2 <= 1
            
            # Create smooth heatmap based on trajectory
            gaussian_grid = np.zeros((grid_size, grid_size))
            
            for i, (val, ar) in enumerate(zip(valence, arousal)):
                # Convert to grid coordinates
                x_idx = int((val + 1) * grid_size / 2)
                y_idx = int((ar + 1) * grid_size / 2)
                
                # Ensure indices are within bounds
                x_idx = max(0, min(grid_size - 1, x_idx))
                y_idx = max(0, min(grid_size - 1, y_idx))
                
                # Add Gaussian blob around the point
                sigma = 3  # Spread of the Gaussian
                for dx in range(-sigma*2, sigma*2 + 1):
                    for dy in range(-sigma*2, sigma*2 + 1):
                        nx, ny = x_idx + dx, y_idx + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            distance = np.sqrt(dx**2 + dy**2)
                            weight = np.exp(-distance**2 / (2 * sigma**2))
                            gaussian_grid[ny, nx] += weight
            
            # Apply circular mask
            gaussian_masked = np.where(circle_mask, gaussian_grid, np.nan)
            
            im = ax1.imshow(gaussian_masked, extent=[-1, 1, -1, 1], origin='lower', cmap='plasma', alpha=0.8)
            
            # Draw movement path
            ax1.plot(valence, arousal, 'white', linewidth=2.5, alpha=0.9, label='Emotion Path')
            ax1.scatter(valence[0], arousal[0], color='lime', s=150, marker='o', 
                       label='Start', edgecolor='black', linewidth=2, zorder=5)
            ax1.scatter(valence[-1], arousal[-1], color='red', s=150, marker='s', 
                       label='End', edgecolor='black', linewidth=2, zorder=5)
            
            # Draw circle boundary
            circle = plt.Circle((0, 0), 1, fill=False, color='white', linewidth=3)
            ax1.add_patch(circle)
            
            # Draw quadrant lines
            ax1.axhline(y=0, color='white', linestyle='-', alpha=0.7, linewidth=2)
            ax1.axvline(x=0, color='white', linestyle='-', alpha=0.7, linewidth=2)
            
            # Add quadrant labels with color-coded backgrounds
            ax1.text(0.6, 0.6, 'EXCITED', fontsize=14, ha='center', va='center', 
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.state_colors['Excited'], 
                             edgecolor='black', linewidth=2, alpha=0.9))
            ax1.text(-0.6, 0.6, 'STRESSED', fontsize=14, ha='center', va='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.state_colors['Stressed'], 
                             edgecolor='black', linewidth=2, alpha=0.9))
            ax1.text(0.6, -0.6, 'CALM', fontsize=14, ha='center', va='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.state_colors['Calm'], 
                             edgecolor='black', linewidth=2, alpha=0.9))
            ax1.text(-0.6, -0.6, 'TIRED', fontsize=14, ha='center', va='center',
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.state_colors['Tired'], 
                             edgecolor='black', linewidth=2, alpha=0.9))
            
            ax1.set_xlabel('Valence (Negative ← → Positive)', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Arousal (Low ← → High)', fontsize=13, fontweight='bold')
            ax1.set_title('�️ Voice Emotional Journey Path', fontsize=16, fontweight='bold', pad=15)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.7)
            cbar.set_label('Movement Intensity', fontsize=11)
            
            ax1.legend(loc='upper left', fontsize=10)
            ax1.set_xlim(-1.1, 1.1)
            ax1.set_ylim(-1.1, 1.1)
            ax1.set_aspect('equal')
        
        # ============ PLOT 2: Summary Statistics Table ============
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        stats_data = []
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            total_samples = len(df)
            
            # Include all states, even with 0 count
            for state in ['Stressed', 'Excited', 'Tired', 'Calm']:
                count = state_counts.get(state, 0)
                proportion = count / total_samples if total_samples > 0 else 0
                stats_data.append([
                    state,
                    f"{count}",
                    f"{proportion:.4f}",
                    f"{proportion*100:.2f}%"
                ])
        
        if stats_data:
            table = ax2.table(cellText=stats_data,
                            colLabels=['State', 'Count', 'Proportion', 'Percentage'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 3)
            
            # Color code state names
            for i, row in enumerate(stats_data, 1):
                state = row[0]
                table[(i, 0)].set_facecolor(self.state_colors.get(state, '#95A5A6'))
                table[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(4):
                table[(0, j)].set_facecolor('#34495E')
                table[(0, j)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('📊 State Summary Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 3: Arousal/Valence Statistics Table ============
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')
        
        dimension_stats = []
        for dim in ['arousal', 'valence']:
            col_name = f'{prefix}{dim}'
            if col_name in df.columns:
                dimension_stats.append([
                    dim.capitalize(),
                    f"{df[col_name].mean():.4f}",
                    f"{df[col_name].std():.4f}",
                    f"{df[col_name].min():.4f}",
                    f"{df[col_name].max():.4f}"
                ])
        
        if dimension_stats:
            table2 = ax3.table(cellText=dimension_stats,
                             colLabels=['Dimension', 'Mean', 'Std Dev', 'Min', 'Max'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1, 3.5)
            
            # Color code dimension names
            table2[(1, 0)].set_facecolor(self.dimension_colors['arousal'])
            table2[(1, 0)].set_text_props(color='white', weight='bold')
            table2[(2, 0)].set_facecolor(self.dimension_colors['valence'])
            table2[(2, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(5):
                table2[(0, j)].set_facecolor('#34495E')
                table2[(0, j)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('📈 Dimension Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # ============ PLOT 4: Average Emotional State Values (Bar Chart) ============
        ax4 = fig.add_subplot(gs[2, 0])
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            state_avgs = state_counts / len(df)  # Convert to proportions
            
            colors = [self.state_colors.get(state, '#95A5A6') for state in state_avgs.index]
            bars = ax4.bar(range(len(state_avgs)), state_avgs.values,
                          color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax4.set_xticks(range(len(state_avgs)))
            ax4.set_xticklabels(state_avgs.index, fontsize=12, fontweight='bold')
            ax4.set_ylabel('Proportion', fontsize=12)
            ax4.set_title('📊 Average Voice State Values', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, val in enumerate(state_avgs.values):
                ax4.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        
        # ============ PLOT 5: State Over Time ============
        ax5 = fig.add_subplot(gs[2, 1:])
        
        if quadrant_col in df.columns:
            # Map states to numeric values for plotting
            state_mapping = {'Stressed': 3, 'Excited': 2, 'Calm': 1, 'Tired': 0}
            numeric_states = df[quadrant_col].map(state_mapping)
            
            # Plot with color segments
            for i in range(len(time_data)-1):
                state = df[quadrant_col].iloc[i]
                color = self.state_colors.get(state, '#95A5A6')
                ax5.plot(time_data.iloc[i:i+2], numeric_states.iloc[i:i+2],
                        color=color, linewidth=4, alpha=0.8)
            
            ax5.set_yticks([0, 1, 2, 3])
            ax5.set_yticklabels(['Tired', 'Calm', 'Excited', 'Stressed'], fontsize=11, fontweight='bold')
            ax5.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            ax5.set_title('📈 Voice Emotional State Over Time', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Add colored background for each state
            ax5.axhspan(2.5, 3.5, facecolor=self.state_colors['Stressed'], alpha=0.15)
            ax5.axhspan(1.5, 2.5, facecolor=self.state_colors['Excited'], alpha=0.15)
            ax5.axhspan(0.5, 1.5, facecolor=self.state_colors['Calm'], alpha=0.15)
            ax5.axhspan(-0.5, 0.5, facecolor=self.state_colors['Tired'], alpha=0.15)
        
        # ============ PLOT 6: State Proportion Pie Chart ============
        ax6 = fig.add_subplot(gs[3, 0])
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            colors_pie = [self.state_colors.get(state, '#95A5A6') for state in state_counts.index]
            labels_pie = [f'{state}\n{count/len(df)*100:.1f}%' 
                         for state, count in state_counts.items()]
            
            ax6.pie(state_counts.values, labels=labels_pie, colors=colors_pie,
                   autopct='', startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
            ax6.set_title('🥧 Voice State Proportion', fontsize=14, fontweight='bold')
        
        # ============ PLOT 7: State Distribution Table ============
        ax7 = fig.add_subplot(gs[3, 1:])
        ax7.axis('off')
        
        if quadrant_col in df.columns:
            state_counts = df[quadrant_col].value_counts()
            table_data = []
            
            # Include all states, even with 0 count
            for state in ['Stressed', 'Excited', 'Tired', 'Calm']:
                count = state_counts.get(state, 0)
                proportion = count / len(df) if len(df) > 0 else 0
                percentage = proportion * 100
                table_data.append([
                    state,
                    f'{count}',
                    f'{proportion:.4f}',
                    f'{percentage:.2f}%'
                ])
            
            table3 = ax7.table(cellText=table_data,
                             colLabels=['State', 'Sample Count', 'Proportion', 'Percentage'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0, 0, 1, 1])
            table3.auto_set_font_size(False)
            table3.set_fontsize(11)
            table3.scale(1, 2.5)
            
            # Color code state names
            for i, row in enumerate(table_data, 1):
                state = row[0]
                table3[(i, 0)].set_facecolor(self.state_colors.get(state, '#95A5A6'))
                table3[(i, 0)].set_text_props(color='white', weight='bold')
            
            # Style header
            for j in range(4):
                table3[(0, j)].set_facecolor('#34495E')
                table3[(0, j)].set_text_props(weight='bold', color='white')
            
            ax7.set_title('📊 Voice State Distribution Table', fontsize=14, fontweight='bold', pad=20)
            
            # Add dominant state indicator
            dominant_state = state_counts.idxmax()
            dominant_count = state_counts.max()
            dominant_pct = (dominant_count / len(df)) * 100
            
            fig.text(0.5, 0.01,
                    f"Most Dominant Voice State: {dominant_state} ({dominant_count} samples - {dominant_pct:.2f}%)",
                    ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor=self.state_colors.get(dominant_state, '#95A5A6'),
                             edgecolor='black', linewidth=2))
        
        # Save
        save_path = f"{base_filename}_voice_states.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Voice States report saved: {save_path}")
        plt.close()
    
    def generate_all_reports(self, df, base_filename):
        """Generate all four voice reports"""
        print(f"\n{'='*80}")
        print("🎤 GENERATING VOICE EMOTION REPORTS")
        print(f"{'='*80}\n")
        
        self.generate_emotion_report(df, base_filename)
        self.generate_acoustic_report(df, base_filename)
        self.generate_dimensions_report(df, base_filename)
        self.generate_states_report(df, base_filename)
        
        print(f"\n{'='*80}")
        print("✅ ALL VOICE REPORTS GENERATED SUCCESSFULLY!")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        base_name = csv_file.replace('.csv', '').replace('_emotion_data', '')
    else:
        csv_file = "angry_emotion_data.csv"
        base_name = "angry"
    
    try:
        df = pd.read_csv(csv_file)
        generator = VoiceReportGenerator()
        generator.generate_all_reports(df, base_name)
    except FileNotFoundError:
        print(f"❌ File not found: {csv_file}")
        print("Usage: python voice_reports.py [csv_file]")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
