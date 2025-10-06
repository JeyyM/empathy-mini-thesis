from emotion_bot import EmotionBot

def main():
    print("=== Quick Webcam Emotion Demo ===")
    print("10-second emotion detection starting in 3 seconds...")
    print("Make sure your face is visible!")
    print()
    
    import time
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("Recording emotions now! Press 'q' to stop early.")
    
    # Create emotion bot and run for 10 seconds
    bot = EmotionBot()
    df = bot.process_webcam(duration_seconds=10, sample_rate=0.5)
    
    # Quick results
    if not df.empty:
        print(f"\nCaptured {len(df)} emotion samples!")
        
        # Show dominant emotion
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        means = {emotion: df[emotion].mean() for emotion in emotions if emotion in df.columns}
        dominant = max(means, key=means.get)
        
        print(f"Your dominant emotion: {dominant.upper()} ({means[dominant]:.1%})")
        
        # Show top 3 emotions
        sorted_emotions = sorted(means.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 3 emotions:")
        for i, (emotion, score) in enumerate(sorted_emotions[:3], 1):
            print(f"{i}. {emotion.capitalize()}: {score:.1%}")
            
    else:
        print("No face detected. Try again with better lighting!")

if __name__ == "__main__":
    main()
