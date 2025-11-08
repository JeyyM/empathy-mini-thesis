#!/usr/bin/env python3
"""
Generate a heatmap showing emotion circle movement patterns from CSV data
"""
import pandas as pd
from emotion_bot import EmotionBot

def create_circle_movement_heatmap_from_csv(csv_file, save_path=None):
    """Create a movement heatmap from existing CSV data"""
    try:
        # Load the data
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} emotion samples from {csv_file}")
        
        # Create an EmotionBot instance and populate with data
        bot = EmotionBot()
        
        # Convert dataframe rows back to emotion_data format
        for _, row in df.iterrows():
            data_point = {
                'timestamp': row['timestamp'],
                'elapsed_seconds': row['elapsed_seconds'],
                'quadrant': row['quadrant'],
                'arousal': row['arousal'],
                'valence': row['valence'],
                'intensity': row['intensity'],
                'excitement': row['excitement'],
                'calmness': row['calmness'],
                'positivity': row['positivity'],
                'negativity': row['negativity'],
                'angry': row['angry'],
                'disgust': row['disgust'],
                'fear': row['fear'],
                'happy': row['happy'],
                'sad': row['sad'],
                'surprise': row['surprise'],
                'neutral': row['neutral']
            }
            bot.emotion_data.append(data_point)
        
        # Generate circle movement heatmap
        bot.plot_circle_movement_heatmap(save_path=save_path)
        
        return True
        
    except Exception as e:
        print(f"Error creating movement heatmap: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "webcam_emotion_data.csv"
    
    print(f"Creating emotion circle movement heatmap from {csv_file}...")
    
    # Create movement heatmap
    save_path = csv_file.replace('.csv', '_movement_heatmap.png')
    success = create_circle_movement_heatmap_from_csv(csv_file, save_path)
    
    if success:
        print(f"\nMovement heatmap saved to {save_path}")
        print("\nThis heatmap shows:")
        print("- Left: Where you spent most time in the emotion circle (hotspots)")
        print("- Right: Your emotional journey path with start/end points")
        print("- Movement statistics and analysis")
    else:
        print("Failed to create movement heatmap")