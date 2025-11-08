#!/usr/bin/env python3
"""
Generate voice prosodic movement heatmap from existing voice emotion data CSV file
Tracks pitch, tempo, volume, and voice quality changes over time
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from voice_emotion_bot import VoiceEmotionBot
import warnings
warnings.filterwarnings("ignore")

def create_voice_movement_heatmap_from_csv(csv_file, save_path=None):
    """Create voice prosodic movement heatmap from existing CSV data"""
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
                'pitch_mean': row.get('pitch_mean', 0),
                'pitch_std': row.get('pitch_std', 0),
                'pitch_range': row.get('pitch_range', 0),
                'pitch_variation': row.get('pitch_variation', 0),
                'volume_mean': row.get('volume_mean', 0),
                'volume_std': row.get('volume_std', 0),
                'volume_range': row.get('volume_range', 0),
                'spectral_centroid': row.get('spectral_centroid', 0),
                'spectral_centroid_std': row.get('spectral_centroid_std', 0),
                'spectral_bandwidth': row.get('spectral_bandwidth', 0),
                'spectral_rolloff': row.get('spectral_rolloff', 0),
                'zero_crossing_rate': row.get('zero_crossing_rate', 0),
                'harmonic_ratio': row.get('harmonic_ratio', 0),
                'voice_tremor': row.get('voice_tremor', 0),
                'silence_ratio': row.get('silence_ratio', 0),
                'speech_rate': row.get('speech_rate', 0),
                'voice_arousal': row.get('voice_arousal', 0),
                'voice_valence': row.get('voice_valence', 0),
                'voice_intensity': row.get('voice_intensity', 0),
                'voice_stress': row.get('voice_stress', 0)
            }
            bot.voice_data.append(data_point)
        
        # Generate voice movement heatmap
        plot_voice_movement_heatmap(bot, save_path=save_path)
        
        return True
        
    except Exception as e:
        print(f"Error creating voice movement heatmap: {e}")
        return False

def plot_voice_movement_heatmap(voice_bot, save_path=None):
    """
    Create a comprehensive voice prosodic movement heatmap
    Shows changes and patterns in pitch, volume, spectral features, and voice quality over time
    """
    if not voice_bot.voice_data:
        print("No voice emotion data available for movement heatmap")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(voice_bot.voice_data)
    
    # Set up the figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Voice Prosodic Movement & Change Analysis', fontsize=16, fontweight='bold')
    
    # Prepare time axis
    time_points = df['time_seconds'].values
    
    # Helper function to calculate movement/change
    def calculate_movement(data, window=5):
        """Calculate movement as rolling standard deviation and changes"""
        if len(data) < window:
            return np.zeros_like(data)
        
        # Calculate rolling standard deviation as measure of variability/movement
        movement = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window//2)
            end_idx = min(len(data), i + window//2 + 1)
            movement[i] = np.std(data[start_idx:end_idx])
        
        return movement
    
    # 1. Pitch Movement Analysis (Top Left)
    pitch_features = ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_variation']
    pitch_data = []
    pitch_labels = []
    
    for feature in pitch_features:
        if feature in df.columns and not df[feature].isna().all():
            # Calculate movement for this feature
            movement = calculate_movement(df[feature].fillna(0).values)
            pitch_data.append(movement)
            pitch_labels.append(feature.replace('_', ' ').title())
    
    if pitch_data:
        # Normalize each feature to 0-1 range for comparison
        normalized_pitch = []
        for data in pitch_data:
            if np.max(data) > np.min(data):
                normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            else:
                normalized = np.zeros_like(data)
            normalized_pitch.append(normalized)
        
        pitch_matrix = np.array(normalized_pitch)
        im1 = axes[0, 0].imshow(pitch_matrix, cmap='YlOrRd', aspect='auto', interpolation='bilinear')
        axes[0, 0].set_title('Pitch Movement & Variation Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Pitch Feature Movement')
        axes[0, 0].set_yticks(range(len(pitch_labels)))
        axes[0, 0].set_yticklabels(pitch_labels)
        
        # Add time labels
        if len(time_points) > 6:
            time_ticks = np.linspace(0, len(time_points)-1, 6, dtype=int)
            axes[0, 0].set_xticks(time_ticks)
            axes[0, 0].set_xticklabels([f'{time_points[i]:.1f}s' for i in time_ticks])
        
        plt.colorbar(im1, ax=axes[0, 0], label='Movement Intensity')
    
    # 2. Volume Movement Analysis (Top Right)
    volume_features = ['volume_mean', 'volume_std', 'volume_range']
    volume_data = []
    volume_labels = []
    
    for feature in volume_features:
        if feature in df.columns and not df[feature].isna().all():
            movement = calculate_movement(df[feature].fillna(0).values)
            volume_data.append(movement)
            volume_labels.append(feature.replace('_', ' ').title())
    
    if volume_data:
        normalized_volume = []
        for data in volume_data:
            if np.max(data) > np.min(data):
                normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            else:
                normalized = np.zeros_like(data)
            normalized_volume.append(normalized)
        
        volume_matrix = np.array(normalized_volume)
        im2 = axes[0, 1].imshow(volume_matrix, cmap='BuPu', aspect='auto', interpolation='bilinear')
        axes[0, 1].set_title('Volume Movement & Dynamics Over Time', fontweight='bold')
        axes[0, 1].set_ylabel('Volume Feature Movement')
        axes[0, 1].set_yticks(range(len(volume_labels)))
        axes[0, 1].set_yticklabels(volume_labels)
        
        if len(time_points) > 6:
            axes[0, 1].set_xticks(time_ticks)
            axes[0, 1].set_xticklabels([f'{time_points[i]:.1f}s' for i in time_ticks])
        
        plt.colorbar(im2, ax=axes[0, 1], label='Movement Intensity')
    
    # 3. Spectral Movement Analysis (Middle Left)
    spectral_features = ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate']
    spectral_data = []
    spectral_labels = []
    
    for feature in spectral_features:
        if feature in df.columns and not df[feature].isna().all():
            movement = calculate_movement(df[feature].fillna(0).values)
            spectral_data.append(movement)
            spectral_labels.append(feature.replace('_', ' ').title())
    
    if spectral_data:
        normalized_spectral = []
        for data in spectral_data:
            if np.max(data) > np.min(data):
                normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            else:
                normalized = np.zeros_like(data)
            normalized_spectral.append(normalized)
        
        spectral_matrix = np.array(normalized_spectral)
        im3 = axes[1, 0].imshow(spectral_matrix, cmap='viridis', aspect='auto', interpolation='bilinear')
        axes[1, 0].set_title('Spectral Feature Movement Over Time', fontweight='bold')
        axes[1, 0].set_ylabel('Spectral Feature Movement')
        axes[1, 0].set_yticks(range(len(spectral_labels)))
        axes[1, 0].set_yticklabels(spectral_labels)
        
        if len(time_points) > 6:
            axes[1, 0].set_xticks(time_ticks)
            axes[1, 0].set_xticklabels([f'{time_points[i]:.1f}s' for i in time_ticks])
        
        plt.colorbar(im3, ax=axes[1, 0], label='Movement Intensity')
    
    # 4. Voice Quality Movement Analysis (Middle Right)
    quality_features = ['harmonic_ratio', 'voice_tremor', 'silence_ratio', 'speech_rate']
    quality_data = []
    quality_labels = []
    
    for feature in quality_features:
        if feature in df.columns and not df[feature].isna().all():
            movement = calculate_movement(df[feature].fillna(0).values)
            quality_data.append(movement)
            quality_labels.append(feature.replace('_', ' ').title())
    
    if quality_data:
        normalized_quality = []
        for data in quality_data:
            if np.max(data) > np.min(data):
                normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            else:
                normalized = np.zeros_like(data)
            normalized_quality.append(normalized)
        
        quality_matrix = np.array(normalized_quality)
        im4 = axes[1, 1].imshow(quality_matrix, cmap='plasma', aspect='auto', interpolation='bilinear')
        axes[1, 1].set_title('Voice Quality Movement Over Time', fontweight='bold')
        axes[1, 1].set_ylabel('Voice Quality Movement')
        axes[1, 1].set_yticks(range(len(quality_labels)))
        axes[1, 1].set_yticklabels(quality_labels)
        
        if len(time_points) > 6:
            axes[1, 1].set_xticks(time_ticks)
            axes[1, 1].set_xticklabels([f'{time_points[i]:.1f}s' for i in time_ticks])
        
        plt.colorbar(im4, ax=axes[1, 1], label='Movement Intensity')
    
    # 5. Overall Prosodic Activity Timeline (Bottom Left)
    if len(df) > 0:
        # Calculate overall prosodic activity as combination of all movements
        overall_activity = np.zeros(len(time_points))
        activity_count = 0
        
        for feature in ['pitch_variation', 'volume_std', 'spectral_centroid_std', 'voice_tremor']:
            if feature in df.columns and not df[feature].isna().all():
                data = df[feature].fillna(0).values
                if np.max(data) > 0:
                    normalized_data = data / np.max(data)
                    overall_activity += normalized_data
                    activity_count += 1
        
        if activity_count > 0:
            overall_activity /= activity_count
            
            axes[2, 0].fill_between(time_points, overall_activity, alpha=0.7, color='green', label='Overall Activity')
            axes[2, 0].plot(time_points, overall_activity, color='darkgreen', linewidth=2)
            
            # Add stress indicator if available
            if 'voice_stress' in df.columns:
                stress_data = df['voice_stress'].fillna(0).values
                axes[2, 0].plot(time_points, stress_data, color='red', alpha=0.8, linewidth=2, label='Voice Stress')
                axes[2, 0].legend()
        
        axes[2, 0].set_title('Overall Prosodic Activity & Stress Timeline', fontweight='bold')
        axes[2, 0].set_xlabel('Time (seconds)')
        axes[2, 0].set_ylabel('Activity Level')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Prosodic Movement Velocity (Bottom Right)
    if len(df) > 1:
        # Calculate velocity of change (first derivative approximation)
        velocity_features = ['pitch_mean', 'volume_mean', 'spectral_centroid']
        
        for i, feature in enumerate(velocity_features):
            if feature in df.columns and not df[feature].isna().all():
                data = df[feature].fillna(0).values
                
                # Calculate velocity (rate of change)
                velocity = np.zeros_like(data)
                for j in range(1, len(data)):
                    velocity[j] = abs(data[j] - data[j-1])
                
                # Normalize for visualization
                if np.max(velocity) > 0:
                    velocity = velocity / np.max(velocity)
                
                color_map = ['blue', 'orange', 'purple']
                axes[2, 1].plot(time_points, velocity, color=color_map[i % len(color_map)], 
                               alpha=0.8, linewidth=2, label=feature.replace('_', ' ').title())
        
        axes[2, 1].set_title('Prosodic Change Velocity Over Time', fontweight='bold')
        axes[2, 1].set_xlabel('Time (seconds)')
        axes[2, 1].set_ylabel('Rate of Change')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Voice movement heatmap saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_voice_movement_heatmap.py <csv_file> [output_file.png]")
        print("Example: python create_voice_movement_heatmap.py voice_emotion_data.csv voice_movement_heatmap.png")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else f"{csv_file.replace('.csv', '_movement_heatmap.png')}"
    
    print(f"🎤 Creating voice movement heatmap from {csv_file}")
    success = create_voice_movement_heatmap_from_csv(csv_file, save_path)
    
    if success:
        print("✅ Voice movement heatmap created successfully!")
    else:
        print("❌ Failed to create voice movement heatmap")