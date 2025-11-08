from emotion_bot import EmotionBot
import os

def main():
    # Create emotion bot instance
    bot = EmotionBot()
    
    # Video file path
    video_path = "face only.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Make sure 'face only.mp4' is in the same folder as this script")
        return
    
    print(f"Processing video: {video_path}")
    print("Analyzing emotions frame by frame...")
    
    # Process the video (sample every 0.5 seconds for better granularity)
    df = bot.process_video(video_path, sample_rate=0.5)
    
    # Show results
    if not df.empty:
        print(f"\nProcessed {len(df)} emotion samples")
        
        # Show summary statistics
        summary = bot.get_emotion_summary()
        print("\nEmotion Summary:")
        for emotion, stats in summary.items():
            print(f"{emotion.capitalize()}: Mean={stats['mean']:.3f}, Max={stats['max']:.3f}")
        
        # Find most dominant emotion overall
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        means = {emotion: df[emotion].mean() for emotion in emotions if emotion in df.columns}
        dominant = max(means, key=means.get)
        print(f"\nMost dominant emotion throughout video: {dominant.upper()} ({means[dominant]:.3f})")
        
        # Plot emotions over time
        print("\nGenerating emotion timeline plot...")
        bot.plot_emotions(save_path="face_only_emotions.png")
        
        # Save data to CSV
        df.to_csv("face_only_emotion_data.csv", index=False)
        print("Data saved to face_only_emotion_data.csv")
        print("Plot saved to face_only_emotions.png")
        
    else:
        print("No faces/emotions detected in the video")

if __name__ == "__main__":
    main()
