from emotion_bot import EmotionBot
import cv2

def main():
    print("=== Webcam Emotion Tracker ===")
    print("This will use your webcam to track emotions in real-time")
    print()
    
    # Get user preferences
    try:
        duration = int(input("Enter duration in seconds (default 30): ") or "30")
    except ValueError:
        duration = 30
    
    try:
        sample_rate = float(input("Enter sample rate in seconds (default 0.5): ") or "0.5")
    except ValueError:
        sample_rate = 0.5
    
    print(f"\nStarting webcam emotion detection for {duration} seconds...")
    print("Controls:")
    print("- Press 'q' to quit early")
    print("- Make sure your face is visible in the camera")
    print()
    
    # Test webcam first
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam")
        print("Make sure your webcam is connected and not being used by another application")
        return
    cap.release()
    
    # Create emotion bot and start tracking
    bot = EmotionBot()
    
    try:
        df = bot.process_webcam(duration_seconds=duration, sample_rate=sample_rate)
        
        # Show results
        if not df.empty:
            print(f"\n=== Results ===")
            print(f"Captured {len(df)} emotion samples over {duration} seconds")
            
            # Show summary statistics
            summary = bot.get_emotion_summary()
            print("\nEmotion Summary (Average intensities):")
            print("-" * 40)
            for emotion, stats in summary.items():
                print(f"{emotion.capitalize():>10}: {stats['mean']:.3f} (max: {stats['max']:.3f})")
            
            # Find most dominant emotion
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            means = {emotion: df[emotion].mean() for emotion in emotions if emotion in df.columns}
            dominant = max(means, key=means.get)
            print(f"\nMost dominant emotion: {dominant.upper()} ({means[dominant]:.3f})")
            
            # Ask if user wants to see the plot
            show_plot = input("\nShow emotion timeline plot? (y/n): ").lower().startswith('y')
            if show_plot:
                print("Generating plot...")
                bot.plot_emotions(save_path="webcam_emotions.png")
                print("Plot saved to webcam_emotions.png")
            
            # Save data
            df.to_csv("webcam_emotion_data.csv", index=False)
            print("Data saved to webcam_emotion_data.csv")
            
        else:
            print("\nNo emotion data was captured")
            print("Possible issues:")
            print("- No face detected in webcam")
            print("- Webcam quality too low")
            print("- Lighting conditions too poor")
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError during emotion tracking: {e}")

if __name__ == "__main__":
    main()
