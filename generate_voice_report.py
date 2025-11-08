#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Voice Emotion Analysis Report Generator

This script creates comprehensive reports from voice emotion data,
including statistical analysis and multiple visualization panels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_voice_report_from_csv(csv_file, save_path=None):
    """
    Create a comprehensive voice emotion analysis report from CSV data
    
    Args:
        csv_file (str): Path to CSV file containing voice emotion data
        save_path (str, optional): Path to save the report image
    """
    try:
        # Load the voice emotion data
        if not Path(csv_file).exists():
            print(f"Error: CSV file '{csv_file}' not found!")
            return False
        
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} voice emotion samples from {csv_file}")
        
        # Validate required columns
        required_cols = ['timestamp', 'start_time', 'end_time']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            # Create dummy timestamps if missing
            if 'timestamp' not in df.columns:
                df['timestamp'] = np.arange(len(df))
        
        # Get emotion columns
        emotion_cols = [col for col in df.columns if col.startswith('voice_') and 
                       col.split('voice_')[1] in ['angry', 'happy', 'sad', 'fear', 'surprise', 'neutral']]
        
        # Calculate time points for plotting
        if 'time_seconds' in df.columns:
            time_points = df['time_seconds'].values
        elif 'timestamp' in df.columns:
            # Convert timestamp strings to seconds if needed
            timestamps = pd.to_datetime(df['timestamp'])
            time_points = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
        elif 'start_time' in df.columns:
            time_points = df['start_time'].values
        else:
            time_points = np.arange(len(df)) * 3  # Assume 3-second intervals
        
        # Calculate duration
        duration_seconds = time_points[-1] - time_points[0] if len(time_points) > 1 else len(df) * 3
        duration_minutes = duration_seconds / 60
        
        # Set up the figure with better layout
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Helper function for emotion statistics
        def get_emotion_stats(emotion_col):
            if emotion_col in df.columns:
                data = df[emotion_col].fillna(0)
                return {
                    'mean': data.mean(),
                    'std': data.std(),
                    'max': data.max(),
                    'min': data.min(),
                    'above_threshold': (data > 0.5).sum() / len(data) * 100
                }
            return {'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'above_threshold': 0}
        
        # 1. Executive Summary (Top Left)
        ax_summary = fig.add_subplot(gs[0:2, 0:2])
        ax_summary.axis('off')
        
        # Find dominant emotion
        emotion_means = {}
        for emotion in emotion_cols:
            if emotion in df.columns:
                emotion_means[emotion.replace('voice_', '')] = df[emotion].mean()
        
        dominant_emotion = max(emotion_means, key=emotion_means.get) if emotion_means else 'unknown'
        dominant_percentage = emotion_means.get(dominant_emotion, 0) * 100
        
        # Calculate average prosodic features
        avg_pitch = df['pitch_mean'].mean() if 'pitch_mean' in df.columns else 0
        avg_volume = df['volume_mean'].mean() if 'volume_mean' in df.columns else 0
        avg_arousal = df['voice_arousal'].mean() if 'voice_arousal' in df.columns else 0
        avg_valence = df['voice_valence'].mean() if 'voice_valence' in df.columns else 0
        
        # Create emotional state classification
        if avg_arousal > 0.2 and avg_valence > 0.2:
            emotional_state = "Excited/Positive"
            state_color = 'green'
        elif avg_arousal > 0.2 and avg_valence < -0.2:
            emotional_state = "Stressed/Agitated"
            state_color = 'red'
        elif avg_arousal < -0.2 and avg_valence > 0.2:
            emotional_state = "Calm/Relaxed"
            state_color = 'blue'
        elif avg_arousal < -0.2 and avg_valence < -0.2:
            emotional_state = "Depressed/Low"
            state_color = 'purple'
        else:
            emotional_state = "Neutral/Balanced"
            state_color = 'gray'
        
        # Summary text
        summary_text = f"""📊 VOICE EMOTION ANALYSIS SUMMARY

🎤 Recording Duration: {duration_minutes:.1f} minutes ({len(df)} samples)

📈 OVERALL EMOTIONAL STATE: {emotional_state.upper()}

🎭 Dominant Voice Emotion: {dominant_emotion.title()} ({dominant_percentage:.1f}%)

📊 Key Metrics:
• Average Arousal: {avg_arousal:.2f} (-1=Low, +1=High)
• Average Valence: {avg_valence:.2f} (-1=Negative, +1=Positive)
• Average Pitch: {avg_pitch:.0f} Hz
• Average Volume: {avg_volume:.3f}

🔍 Voice Characteristics:"""
        
        # Add conditional characteristics based on available columns
        if 'pitch_std' in df.columns:
            pitch_variation = "Varied" if df['pitch_std'].mean() > 20 else "Steady"
            summary_text += f"\n• Speech Pattern: {pitch_variation}"
        
        if 'harmonic_ratio' in df.columns:
            voice_quality = "Clear" if df['harmonic_ratio'].mean() > 0.7 else "Rough"
            summary_text += f"\n• Voice Quality: {voice_quality}"
        
        if 'voice_stress' in df.columns:
            stress_level = "High" if df['voice_stress'].mean() > 0.6 else "Moderate" if df['voice_stress'].mean() > 0.3 else "Low"
            summary_text += f"\n• Stress Level: {stress_level}"
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=1", facecolor=state_color, alpha=0.1))
        
        # 2. Emotion Distribution Pie Chart (Top Right)
        ax_pie = fig.add_subplot(gs[0, 2:4])
        
        if emotion_means:
            emotions = list(emotion_means.keys())
            values = list(emotion_means.values())
            
            # Filter out very small values for cleaner chart
            filtered_emotions = [(e, v) for e, v in zip(emotions, values) if v > 0.05]
            
            if filtered_emotions:
                emotions, values = zip(*filtered_emotions)
                colors = ['red', 'yellow', 'blue', 'purple', 'orange', 'brown', 'gray']
                
                ax_pie.pie(values, labels=[e.title() for e in emotions], autopct='%1.1f%%',
                          colors=colors[:len(emotions)], startangle=90)
                ax_pie.set_title('Voice Emotion Distribution', fontweight='bold', fontsize=14)
        
        # 3. Arousal-Valence Trajectory (Middle Left)
        ax_trajectory = fig.add_subplot(gs[1, 2:4])
        
        if 'voice_arousal' in df.columns and 'voice_valence' in df.columns:
            arousal = df['voice_arousal'].fillna(0)
            valence = df['voice_valence'].fillna(0)
            
            # Create color map based on time
            colors = time_points
            scatter = ax_trajectory.scatter(valence, arousal, c=colors, cmap='viridis', alpha=0.7, s=30)
            
            # Draw quadrant lines
            ax_trajectory.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax_trajectory.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax_trajectory.text(0.7, 0.7, 'Excited', ha='center', va='center', fontweight='bold', 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            ax_trajectory.text(-0.7, 0.7, 'Stressed', ha='center', va='center', fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
            ax_trajectory.text(-0.7, -0.7, 'Depressed', ha='center', va='center', fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
            ax_trajectory.text(0.7, -0.7, 'Calm', ha='center', va='center', fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7))
            
            ax_trajectory.set_xlabel('Valence (Negative ← → Positive)')
            ax_trajectory.set_ylabel('Arousal (Low ← → High)')
            ax_trajectory.set_title('Voice Emotional Journey (Arousal vs Valence)', fontweight='bold')
            ax_trajectory.set_xlim(-1.1, 1.1)
            ax_trajectory.set_ylim(-1.1, 1.1)
            ax_trajectory.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax_trajectory, label='Time (seconds)')
        
        # 4. Voice Prosodic Features Timeline (Middle)
        ax_prosodic = fig.add_subplot(gs[2, 0:2])
        
        prosodic_features = {
            'Pitch': ('pitch_mean', 'blue'),
            'Volume': ('volume_mean', 'orange'),
            'Spectral Centroid': ('spectral_centroid', 'green'),
            'Speech Rate': ('speech_rate', 'purple')
        }
        
        plotted_features = 0
        for label, (feature, color) in prosodic_features.items():
            if feature in df.columns:
                data = df[feature].fillna(0)
                # Normalize for comparison
                if data.max() > data.min():
                    normalized_data = (data - data.min()) / (data.max() - data.min())
                else:
                    normalized_data = data
                ax_prosodic.plot(time_points, normalized_data, label=label, color=color, linewidth=2, alpha=0.8)
                plotted_features += 1
        
        if plotted_features > 0:
            ax_prosodic.set_xlabel('Time (seconds)')
            ax_prosodic.set_ylabel('Normalized Intensity')
            ax_prosodic.set_title('Voice Prosodic Features Over Time', fontweight='bold')
            ax_prosodic.legend()
            ax_prosodic.grid(True, alpha=0.3)
        else:
            ax_prosodic.text(0.5, 0.5, 'Prosodic features not available', 
                           transform=ax_prosodic.transAxes, ha='center', va='center',
                           fontsize=12, style='italic')
        
        # 5. Emotion Intensity Timeline (Middle Right)
        ax_intensity = fig.add_subplot(gs[2, 2:4])
        
        # Plot main emotional dimensions
        if 'voice_intensity' in df.columns:
            ax_intensity.fill_between(time_points, df['voice_intensity'].fillna(0), 
                                     alpha=0.6, color='orange', label='Voice Intensity')
        
        if 'voice_stress' in df.columns:
            ax_intensity.plot(time_points, df['voice_stress'].fillna(0), 
                             color='red', linewidth=2, alpha=0.8, label='Voice Stress')
        
        if 'voice_arousal' in df.columns:
            arousal_norm = (df['voice_arousal'].fillna(0) + 1) / 2  # Convert from -1,1 to 0,1
            ax_intensity.plot(time_points, arousal_norm, 
                             color='purple', linewidth=2, alpha=0.8, label='Arousal')
        
        ax_intensity.set_xlabel('Time (seconds)')
        ax_intensity.set_ylabel('Intensity Level')
        ax_intensity.set_title('Voice Emotional Intensity Over Time', fontweight='bold')
        ax_intensity.legend()
        ax_intensity.grid(True, alpha=0.3)
        ax_intensity.set_ylim(0, 1)
        
        # 6. Emotion Statistics Bar Chart (Bottom Left)
        ax_stats = fig.add_subplot(gs[3, 0:2])
        
        emotion_stats = {}
        for emotion in ['angry', 'happy', 'sad', 'fear', 'surprise', 'neutral']:
            col = f'voice_{emotion}'
            stats = get_emotion_stats(col)
            emotion_stats[emotion] = stats['mean']
        
        if emotion_stats:
            emotions = list(emotion_stats.keys())
            means = list(emotion_stats.values())
            
            bars = ax_stats.bar(emotions, means, 
                               color=['red', 'yellow', 'blue', 'purple', 'orange', 'brown'])
            
            # Add value labels on bars
            for bar, value in zip(bars, means):
                height = bar.get_height()
                ax_stats.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax_stats.set_xlabel('Voice Emotion')
            ax_stats.set_ylabel('Average Intensity')
            ax_stats.set_title('Average Voice Emotion Intensities', fontweight='bold')
            ax_stats.set_ylim(0, max(means) * 1.2 if means else 1)
            
            # Rotate labels for better readability
            plt.setp(ax_stats.get_xticklabels(), rotation=45, ha='right')
        
        # 7. Voice Quality Analysis (Bottom Right)
        ax_quality = fig.add_subplot(gs[3, 2:4])
        
        quality_metrics = {
            'Harmonic Ratio': ('harmonic_ratio', 'green'),
            'Voice Tremor': ('voice_tremor', 'red'),
            'Silence Ratio': ('silence_ratio', 'blue'),
            'Speech Rate': ('speech_rate', 'purple')
        }
        
        quality_values = []
        quality_labels = []
        quality_colors = []
        
        for label, (feature, color) in quality_metrics.items():
            if feature in df.columns:
                value = df[feature].fillna(0).mean()
                quality_values.append(value)
                quality_labels.append(label)
                quality_colors.append(color)
        
        if quality_values:
            # Normalize values to 0-1 range for radar chart
            max_val = max(quality_values) if max(quality_values) > 0 else 1
            normalized_values = [v/max_val for v in quality_values]
            
            bars = ax_quality.barh(quality_labels, normalized_values, color=quality_colors, alpha=0.7)
            
            # Add value labels
            for i, (bar, value, orig_value) in enumerate(zip(bars, normalized_values, quality_values)):
                ax_quality.text(value + 0.02, i, f'{orig_value:.3f}', 
                              va='center', fontweight='bold')
            
            ax_quality.set_xlabel('Normalized Quality Score')
            ax_quality.set_title('Voice Quality Metrics', fontweight='bold')
            ax_quality.set_xlim(0, 1.2)
            ax_quality.grid(True, alpha=0.3, axis='x')
        else:
            ax_quality.text(0.5, 0.5, 'Voice quality metrics not available', 
                          transform=ax_quality.transAxes, ha='center', va='center',
                          fontsize=12, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Voice emotion report saved to: {save_path}")
        
        plt.show()
        
        # Print detailed statistics
        print("\n" + "="*60)
        print("DETAILED VOICE EMOTION ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 RECORDING INFORMATION:")
        print(f"Duration: {duration_minutes:.2f} minutes")
        print(f"Total segments analyzed: {len(df)}")
        print(f"Sampling rate: {len(df)/duration_minutes:.1f} samples per minute")
        
        print(f"\n🎭 EMOTION ANALYSIS:")
        for emotion in ['angry', 'happy', 'sad', 'fear', 'surprise', 'neutral']:
            col = f'voice_{emotion}'
            stats = get_emotion_stats(col)
            print(f"{emotion.title():12}: Mean={stats['mean']:.3f}, Max={stats['max']:.3f}, "
                  f"Above threshold={stats['above_threshold']:.1f}% of time")
        
        print(f"\n📈 PROSODIC ANALYSIS:")
        prosodic_cols = ['pitch_mean', 'pitch_std', 'volume_mean', 'spectral_centroid']
        for col in prosodic_cols:
            if col in df.columns:
                data = df[col].fillna(0)
                print(f"{col.replace('_', ' ').title():18}: Mean={data.mean():.2f}, "
                      f"Std={data.std():.2f}, Range={data.max()-data.min():.2f}")
            else:
                print(f"{col.replace('_', ' ').title():18}: Not available")
        
        print(f"\n🔍 VOICE QUALITY:")
        quality_cols = ['harmonic_ratio', 'voice_tremor', 'silence_ratio', 'speech_rate']
        for col in quality_cols:
            if col in df.columns:
                data = df[col].fillna(0)
                print(f"{col.replace('_', ' ').title():15}: Mean={data.mean():.3f}, "
                      f"Std={data.std():.3f}")
            else:
                print(f"{col.replace('_', ' ').title():15}: Not available")
        
        print("\n" + "="*60)
        print("INTERPRETATION GUIDE:")
        print("- Arousal: -1 (very low energy) to +1 (very high energy)")
        print("- Valence: -1 (very negative) to +1 (very positive)")
        print("- Emotions: 0 (not present) to 1 (strongly present)")
        print("- Harmonic Ratio: 0 (rough voice) to 1 (clear voice)")
        print("- Voice Tremor: 0 (steady) to higher values (shaky)")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating voice emotion report: {str(e)}")
        return False

def main():
    """Example usage of voice emotion report generator"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_voice_report.py <csv_file> [output_file.png]")
        print("Example: python generate_voice_report.py voice_emotion_data.csv voice_report.png")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else f"{csv_file.replace('.csv', '_report.png')}"
    
    print(f"🎤 Creating voice emotion report from {csv_file}")
    success = create_voice_report_from_csv(csv_file, save_path)
    
    if success:
        print("✅ Voice emotion report created successfully!")
    else:
        print("❌ Failed to create voice emotion report")

if __name__ == "__main__":
    main()