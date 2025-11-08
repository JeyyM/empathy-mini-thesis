#!/usr/bin/env python3
"""
Generate an easy-to-understand emotion report from existing CSV data
"""
import pandas as pd
from emotion_bot import EmotionBot

def create_layperson_report_from_csv(csv_file, save_path=None):
    """Create a user-friendly report from existing CSV data"""
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
        
        # Generate layperson report
        bot.generate_layperson_report(save_path=save_path)
        
        return True
        
    except Exception as e:
        print(f"Error creating report: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "webcam_emotion_data.csv"
    
    print(f"Creating easy-to-read emotion report from {csv_file}...")
    
    # Create report
    save_path = csv_file.replace('.csv', '_report.png')
    success = create_layperson_report_from_csv(csv_file, save_path)
    
    if success:
        print(f"Report saved to {save_path}")
        print("\nThis report is designed to be easily understood by anyone!")
        print("It shows:")
        print("- Overall mood breakdown")
        print("- Energy and mood over time") 
        print("- Time spent in different emotional states")
        print("- Key insights and recommendations")
        print("- Your emotional journey timeline")
    else:
        print("Failed to create report")