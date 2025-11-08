from emotion_bot import EmotionBot
import os

def main():
    print("=== Emotion Tracker ===")
    print("Enter 'camera' for webcam mode")
    print("Or enter a video filename to process that file")
    print()
    
    user_input = input("Input (camera or filename): ").strip()
    
    # Create emotion bot instance
    bot = EmotionBot()
    
    if user_input.lower() == "camera":
        print("\n=== Webcam Mode ===")
        
        # Get webcam settings
        try:
            duration = int(input("Duration in seconds (default 30): ") or "30")
        except ValueError:
            duration = 30
            
        try:
            sample_rate = float(input("Sample rate in seconds (default 0.5): ") or "0.5")
        except ValueError:
            sample_rate = 0.5
        
        print(f"\nStarting webcam for {duration} seconds...")
        print("Press 'q' to quit early")
        
        # Test webcam access
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return
        cap.release()
        
        # Process webcam
        df = bot.process_webcam(duration_seconds=duration, sample_rate=sample_rate)
        output_prefix = "webcam"
        
    else:
        print(f"\n=== Video File Mode ===")
        video_path = user_input
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: File '{video_path}' not found")
            print("Make sure the file is in the current directory or provide full path")
            return
        
        # Get video settings
        try:
            sample_rate = float(input("Sample rate in seconds (default 1.0): ") or "1.0")
        except ValueError:
            sample_rate = 1.0
        
        print(f"Processing video: {video_path}")
        
        # Process video
        df = bot.process_video(video_path, sample_rate=sample_rate)
        output_prefix = os.path.splitext(video_path)[0].replace(" ", "_")
    
    # Show results
    if not df.empty:
        print(f"\n=== Results ===")
        print(f"Processed {len(df)} emotion samples")
        
        # Show summary statistics
        summary = bot.get_emotion_summary()
        print("\nEmotion Summary:")
        print("-" * 50)
        for emotion, stats in summary.items():
            print(f"{emotion.capitalize():>10}: Mean={stats['mean']:.3f}, Max={stats['max']:.3f}")
        
        # Find most dominant emotion
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        means = {emotion: df[emotion].mean() for emotion in emotions if emotion in df.columns}
        dominant = max(means, key=means.get)
        print(f"\nMost dominant emotion: {dominant.upper()} ({means[dominant]:.3f})")
        
        # Ask about visualization options
        print("\nVisualization options:")
        print("1. Line plots (technical)")
        print("2. Technical heatmaps")
        print("3. Circle movement heatmap")
        print("4. Easy-to-read report (for everyone)")
        print("5. All visualizations")
        viz_choice = input("Choose visualization (1/2/3/4/5): ").strip() or "4"
        
        # Ask about saving results
        save_results = input("\nSave results? (y/n): ").lower().startswith('y')
        
        if save_results:
            # Save data first
            csv_filename = f"{output_prefix}_emotion_data.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
            
            # Generate and save visualizations
            if viz_choice in ["1", "5"]:
                plot_filename = f"{output_prefix}_emotions.png"
                bot.plot_emotions(save_path=plot_filename)
                print(f"Technical line plot saved to {plot_filename}")
            
            if viz_choice in ["2", "5"]:
                heatmap_filename = f"{output_prefix}_heatmap.png"
                bot.plot_heatmap(save_path=heatmap_filename)
                print(f"Technical heatmap saved to {heatmap_filename}")
            
            if viz_choice in ["3", "5"]:
                movement_filename = f"{output_prefix}_movement_heatmap.png"
                bot.plot_circle_movement_heatmap(save_path=movement_filename)
                print(f"Circle movement heatmap saved to {movement_filename}")
            
            if viz_choice in ["4", "5"]:
                report_filename = f"{output_prefix}_report.png"
                bot.generate_layperson_report(save_path=report_filename)
                print(f"Easy-to-read report saved to {report_filename}")
        else:
            # Just show the plots
            if viz_choice in ["1", "5"]:
                bot.plot_emotions()
            
            if viz_choice in ["2", "5"]:
                bot.plot_heatmap()
            
            if viz_choice in ["3", "5"]:
                bot.plot_circle_movement_heatmap()
            
            if viz_choice in ["4", "5"]:
                bot.generate_layperson_report()
        
    else:
        print("\nNo emotion data was captured")
        if user_input.lower() == "camera":
            print("Try with better lighting or make sure your face is visible")
        else:
            print("No faces detected in the video file")

if __name__ == "__main__":
    main()
