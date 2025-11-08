#!/usr/bin/env python3
"""
Generate voice emotion intensity heatmap from existing voice emotion data CSV file
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from voice_emotion_bot import VoiceEmotionBot
import warnings
warnings.filterwarnings("ignore")

def create_voice_heatmap_from_csv(csv_file, save_path=None):
    """Create voice emotion intensity heatmap from existing CSV data"""
    try:
        # Load the data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} voice emotion samples from {csv_file}")
        
        # Create a VoiceEmotionBot instance and populate with data
        bot = VoiceEmotionBot()
        
        # Convert dataframe rows back to voice_data format
        for _, row in df.iterrows():
            data_point = {
                'timestamp': pd.to_datetime(row['timestamp']) if 'timestamp' in row else None,
                'time_seconds': row['time_seconds'],
                'voice_arousal': row.get('voice_arousal', 0),
                'voice_valence': row.get('voice_valence', 0),
                'voice_intensity': row.get('voice_intensity', 0),
                'voice_stress': row.get('voice_stress', 0),
                'voice_angry': row.get('voice_angry', 0),
                'voice_disgust': row.get('voice_disgust', 0),
                'voice_fear': row.get('voice_fear', 0),
                'voice_happy': row.get('voice_happy', 0),
                'voice_sad': row.get('voice_sad', 0),
                'voice_surprise': row.get('voice_surprise', 0),
                'voice_neutral': row.get('voice_neutral', 0),
                'pitch_mean': row.get('pitch_mean', 0),
                'volume_mean': row.get('volume_mean', 0)
            }
            bot.voice_data.append(data_point)
        
        # Generate voice heatmap
        plot_voice_heatmap(bot, save_path=save_path)
        
        return True
        
    except Exception as e:
        print(f"Error creating voice heatmap: {e}")
        return False

def plot_voice_heatmap(voice_bot, save_path=None):
    """
    Create a comprehensive voice emotion intensity heatmap
    Shows voice arousal, valence, intensity, and emotion patterns over time
    """
    if not voice_bot.voice_data:
        print("No voice emotion data available for heatmap")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(voice_bot.voice_data)
    
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Voice Emotion Intensity Heatmap Analysis', fontsize=16, fontweight='bold')
    
    # Prepare time axis
    time_points = df['time_seconds'].values
    
    # 1. Voice Emotional Dimensions Heatmap (Top Left)
    dimensions_data = []
    dimension_labels = []
    
    for col in ['voice_arousal', 'voice_valence', 'voice_intensity', 'voice_stress']:
        if col in df.columns:
            dimensions_data.append(df[col].values)
            dimension_labels.append(col.replace('voice_', '').title())
    
    if dimensions_data:
        dimensions_matrix = np.array(dimensions_data)
        im1 = axes[0, 0].imshow(dimensions_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
        axes[0, 0].set_title('Voice Emotional Dimensions Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Emotional Dimension')
        axes[0, 0].set_yticks(range(len(dimension_labels)))
        axes[0, 0].set_yticklabels(dimension_labels)
        
        # Add time labels
        time_ticks = np.linspace(0, len(time_points)-1, 6, dtype=int)
        axes[0, 0].set_xticks(time_ticks)
        axes[0, 0].set_xticklabels([f'{time_points[i]:.1f}s' for i in time_ticks])
        
        plt.colorbar(im1, ax=axes[0, 0], label='Intensity (-1 to 1)')
    
    # 2. Basic Voice Emotions Heatmap (Top Right)
    emotion_data = []
    emotion_labels = []
    
    basic_emotions = ['voice_angry', 'voice_happy', 'voice_sad', 'voice_fear', 
                     'voice_surprise', 'voice_disgust', 'voice_neutral']
    
    for emotion in basic_emotions:
        if emotion in df.columns:
            emotion_data.append(df[emotion].values)
            emotion_labels.append(emotion.replace('voice_', '').title())
    
    if emotion_data:
        emotion_matrix = np.array(emotion_data)
        im2 = axes[0, 1].imshow(emotion_matrix, cmap='viridis', aspect='auto', interpolation='bilinear')
        axes[0, 1].set_title('Voice Emotion Categories Over Time', fontweight='bold')
        axes[0, 1].set_ylabel('Emotion Type')
        axes[0, 1].set_yticks(range(len(emotion_labels)))
        axes[0, 1].set_yticklabels(emotion_labels)
        
        # Add time labels
        axes[0, 1].set_xticks(time_ticks)
        axes[0, 1].set_xticklabels([f'{time_points[i]:.1f}s' for i in time_ticks])
        
        plt.colorbar(im2, ax=axes[0, 1], label='Intensity (0 to 1)')
    
    # 3. Voice Prosodic Features Heatmap (Middle Left)
    prosodic_data = []
    prosodic_labels = []
    
    for col in ['pitch_mean', 'volume_mean', 'spectral_centroid']:
        if col in df.columns:
            # Normalize the data for better visualization
            normalized_data = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6)
            prosodic_data.append(normalized_data.values)
            prosodic_labels.append(col.replace('_', ' ').title())
    
    if prosodic_data:
        prosodic_matrix = np.array(prosodic_data)
        im3 = axes[1, 0].imshow(prosodic_matrix, cmap='plasma', aspect='auto', interpolation='bilinear')
        axes[1, 0].set_title('Voice Prosodic Features Over Time', fontweight='bold')
        axes[1, 0].set_ylabel('Prosodic Feature')
        axes[1, 0].set_yticks(range(len(prosodic_labels)))
        axes[1, 0].set_yticklabels(prosodic_labels)
        
        # Add time labels
        axes[1, 0].set_xticks(time_ticks)
        axes[1, 0].set_xticklabels([f'{time_points[i]:.1f}s' for i in time_ticks])
        
        plt.colorbar(im3, ax=axes[1, 0], label='Normalized Intensity')
    
    # 4. Arousal-Valence Space Density (Middle Right)
    if 'voice_arousal' in df.columns and 'voice_valence' in df.columns:
        # Create 2D histogram for arousal-valence space
        arousal = df['voice_arousal'].values
        valence = df['voice_valence'].values
        
        # Filter out invalid values
        valid_mask = ~(np.isnan(arousal) | np.isnan(valence))
        arousal = arousal[valid_mask]
        valence = valence[valid_mask]
        
        if len(arousal) > 0:
            hist, x_edges, y_edges = np.histogram2d(valence, arousal, bins=20, range=[[-1, 1], [-1, 1]])
            im4 = axes[1, 1].imshow(hist, cmap='hot', origin='lower', extent=[-1, 1, -1, 1])
            axes[1, 1].set_title('Voice Arousal-Valence Density Map', fontweight='bold')
            axes[1, 1].set_xlabel('Valence (Negative ← → Positive)')
            axes[1, 1].set_ylabel('Arousal (Low ← → High)')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add quadrant lines
            axes[1, 1].axhline(y=0, color='white', linestyle='--', alpha=0.7)
            axes[1, 1].axvline(x=0, color='white', linestyle='--', alpha=0.7)
            
            # Add quadrant labels
            axes[1, 1].text(0.5, 0.5, 'Excited', ha='center', va='center', color='white', fontweight='bold')
            axes[1, 1].text(-0.5, 0.5, 'Stressed', ha='center', va='center', color='white', fontweight='bold')
            axes[1, 1].text(-0.5, -0.5, 'Depressed', ha='center', va='center', color='white', fontweight='bold')
            axes[1, 1].text(0.5, -0.5, 'Calm', ha='center', va='center', color='white', fontweight='bold')
            
            plt.colorbar(im4, ax=axes[1, 1], label='Frequency')
    
    # 5. Voice Intensity Timeline (Bottom Left)
    if 'voice_intensity' in df.columns:
        axes[2, 0].fill_between(time_points, df['voice_intensity'], alpha=0.7, color='orange', label='Voice Intensity')
        if 'voice_stress' in df.columns:
            axes[2, 0].plot(time_points, df['voice_stress'], color='red', alpha=0.8, linewidth=2, label='Voice Stress')
        
        axes[2, 0].set_title('Voice Intensity & Stress Over Time', fontweight='bold')
        axes[2, 0].set_xlabel('Time (seconds)')
        axes[2, 0].set_ylabel('Intensity Level')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Dominant Voice Emotion Timeline (Bottom Right)
    if emotion_data:
        # Find dominant emotion at each timepoint
        dominant_emotions = np.argmax(emotion_matrix, axis=0)
        
        # Create color map for emotions
        emotion_colors = ['red', 'yellow', 'blue', 'purple', 'orange', 'brown', 'gray']
        
        # Plot dominant emotion over time
        for i, (emotion_idx, color) in enumerate(zip(dominant_emotions, emotion_colors * len(time_points))):
            if i < len(time_points):
                axes[2, 1].scatter(time_points[i], emotion_idx, c=emotion_colors[emotion_idx % len(emotion_colors)], 
                                 s=50, alpha=0.7)
        
        axes[2, 1].set_title('Dominant Voice Emotion Over Time', fontweight='bold')
        axes[2, 1].set_xlabel('Time (seconds)')
        axes[2, 1].set_ylabel('Dominant Emotion')
        axes[2, 1].set_yticks(range(len(emotion_labels)))
        axes[2, 1].set_yticklabels(emotion_labels)
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Voice emotion heatmap saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_voice_heatmap.py <csv_file> [output_file.png]")
        print("Example: python create_voice_heatmap.py voice_emotion_data.csv voice_heatmap.png")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else f"{csv_file.replace('.csv', '_heatmap.png')}"
    
    print(f"🎤 Creating voice emotion heatmap from {csv_file}")
    success = create_voice_heatmap_from_csv(csv_file, save_path)
    
    if success:
        print("✅ Voice emotion heatmap created successfully!")
    else:
        print("❌ Failed to create voice emotion heatmap")