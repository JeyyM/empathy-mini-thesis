from emotion_bot import EmotionBot
from unified_emotion_tracker import UnifiedEmotionTracker
from live_unified_tracker import LiveUnifiedTracker
import os

def main():
    print("=== Unified Emotion Tracker (Facial + Voice) ===")
    print("Enter 'camera' for webcam mode (LIVE facial + voice)")
    print("Or enter a video filename to process that file")
    print()
    
    user_input = input("Input (camera or filename): ").strip()
    
    if user_input.lower() == "camera":
        # Use live tracker for webcam (facial + voice simultaneously)
        live_tracker = LiveUnifiedTracker()
        bot = live_tracker.face_bot  # Keep reference for compatibility
    else:
        # Use unified tracker for video files
        tracker = UnifiedEmotionTracker()
        bot = tracker.face_bot  # Keep reference for compatibility
    
    if user_input.lower() == "camera":
        print("\n=== Live Webcam Mode (Facial + Voice) ===")
        
        # Get webcam settings
        try:
            duration = int(input("Duration in seconds (default 30): ") or "30")
        except ValueError:
            duration = 30
            
        try:
            sample_rate = float(input("Sample rate in seconds (default 1.0): ") or "1.0")
        except ValueError:
            sample_rate = 1.0
        
        print(f"\nStarting LIVE capture for {duration} seconds...")
        print("📸 Webcam will track your facial expressions")
        print("🎤 Microphone will capture your voice")
        print("Press 'q' to quit early")
        
        # Test webcam access
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return
        cap.release()
        
        # Process live (facial + voice simultaneously)
        df = live_tracker.process_live(duration_seconds=duration, sample_rate=sample_rate)
        output_prefix = "webcam_live"
        has_voice_data = 'voice_arousal' in df.columns or 'voice_angry' in df.columns
        
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
        print("This will analyze BOTH facial expressions AND voice emotions")
        
        # Process video with unified tracker (facial + voice)
        df = tracker.process_video_unified(video_path, sample_interval=sample_rate)
        output_prefix = os.path.splitext(video_path)[0].replace(" ", "_")
        has_voice_data = 'voice_arousal' in df.columns or 'voice_angry' in df.columns
    
    # Show results
    if not df.empty:
        print(f"\n=== Results ===")
        print(f"Processed {len(df)} emotion samples")
        
        # Check if this is unified data (has voice columns) or facial only
        if 'has_voice_data' not in locals():
            has_voice_data = 'voice_arousal' in df.columns or 'voice_angry' in df.columns
        
        # Show summary statistics
        if not has_voice_data:
            # Facial only summary
            summary = bot.get_emotion_summary()
            print("\n📸 Facial Emotion Summary:")
            print("-" * 50)
            for emotion, stats in summary.items():
                print(f"{emotion.capitalize():>10}: Mean={stats['mean']:.3f}, Max={stats['max']:.3f}")
            
            # Find most dominant facial emotion
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            means = {emotion: df[emotion].mean() for emotion in emotions if emotion in df.columns}
            dominant = max(means, key=means.get)
            print(f"\nMost dominant facial emotion: {dominant.upper()} ({means[dominant]:.3f})")
            
        else:
            # Unified summary (facial + voice)
            print("\n📊 UNIFIED EMOTION ANALYSIS (Facial + Voice)")
            print("=" * 60)
            
            # Facial emotions
            print("\n📸 FACIAL Emotions:")
            print("-" * 50)
            facial_emotions = ['facial_angry', 'facial_disgust', 'facial_fear', 
                              'facial_happy', 'facial_sad', 'facial_surprise', 'facial_neutral']
            for emotion in facial_emotions:
                if emotion in df.columns:
                    emotion_name = emotion.replace('facial_', '').capitalize()
                    print(f"{emotion_name:>10}: Mean={df[emotion].mean():.3f}, Max={df[emotion].max():.3f}")
            
            # Voice emotions
            print("\n🎤 VOICE Emotions:")
            print("-" * 50)
            voice_emotions = ['voice_angry', 'voice_disgust', 'voice_fear',
                             'voice_happy', 'voice_sad', 'voice_surprise', 'voice_neutral']
            for emotion in voice_emotions:
                if emotion in df.columns:
                    emotion_name = emotion.replace('voice_', '').capitalize()
                    print(f"{emotion_name:>10}: Mean={df[emotion].mean():.3f}, Max={df[emotion].max():.3f}")
            
            # Combined metrics
            print("\n🔄 COMBINED Metrics:")
            print("-" * 50)
            if 'combined_arousal' in df.columns:
                print(f"  Arousal: {df['combined_arousal'].mean():.3f}")
            if 'combined_valence' in df.columns:
                print(f"  Valence: {df['combined_valence'].mean():.3f}")
            if 'combined_intensity' in df.columns:
                print(f" Intensity: {df['combined_intensity'].mean():.3f}")
            
            # Dominant emotions
            facial_means = {e: df[f'facial_{e}'].mean() for e in ['angry', 'happy', 'sad', 'fear'] 
                          if f'facial_{e}' in df.columns}
            voice_means = {e: df[f'voice_{e}'].mean() for e in ['angry', 'happy', 'sad', 'fear']
                          if f'voice_{e}' in df.columns}
            
            if facial_means:
                dom_facial = max(facial_means, key=facial_means.get)
                print(f"\n  Most dominant FACIAL: {dom_facial.upper()} ({facial_means[dom_facial]:.3f})")
            if voice_means:
                dom_voice = max(voice_means, key=voice_means.get)
                print(f"  Most dominant VOICE:  {dom_voice.upper()} ({voice_means[dom_voice]:.3f})")

        
        # Ask about visualization options
        print("\n📊 Visualization options:")
        if has_voice_data:
            print("1. Unified analysis (facial + voice combined)")
            print("2. Facial emotions line plot")
            print("3. Voice features line plot")
            print("4. Facial emotion heatmap")
            print("5. Voice emotion heatmap")
            print("6. Facial movement heatmap (circle)")
            print("7. Voice movement heatmap (circle)")
            print("8. Easy-to-read report (for everyone)")
            print("9. All standard visualizations (1-8)")
            print("10. 🌟 Comprehensive reports (ALL data - facial & voice separate)")
            print("11. 🔥 EVERYTHING (all standard + comprehensive)")
            print("12. 🚀 ULTIMATE MEGA REPORT (single massive 24-subplot analysis)")
            viz_choice = input("Choose visualization (1-12): ").strip() or "1"
        else:
            print("1. Line plots (technical)")
            print("2. Technical heatmaps")
            print("3. Circle movement heatmap")
            print("4. Easy-to-read report (for everyone)")
            print("5. All standard visualizations")
            print("6. 🌟 Comprehensive report (ALL facial data)")
            viz_choice = input("Choose visualization (1/2/3/4/5/6): ").strip() or "4"
        
        # Ask about saving results
        save_results = input("\nSave results? (y/n): ").lower().startswith('y')
        
        if save_results:
            # Save data first
            csv_filename = f"{output_prefix}_emotion_data.csv"
            df.to_csv(csv_filename, index=False)
            print(f"✅ Data saved to {csv_filename}")
            
            # Generate and save visualizations
            if has_voice_data:
                # Unified (facial + voice) visualizations
                if viz_choice in ["1", "9", "11"]:
                    unified_filename = f"{output_prefix}_unified_emotions.png"
                    # Use appropriate tracker
                    if user_input.lower() == "camera":
                        # For live mode, create simple visualization
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                        
                        time_data = df['time_seconds']
                        ax[0].plot(time_data, df['facial_arousal'], label='Facial Arousal', linewidth=2)
                        ax[0].plot(time_data, df['facial_valence'], label='Facial Valence', linewidth=2)
                        if 'voice_arousal' in df.columns:
                            ax[0].plot(time_data, df['voice_arousal'], label='Voice Arousal', linewidth=2, linestyle='--')
                            ax[0].plot(time_data, df['voice_valence'], label='Voice Valence', linewidth=2, linestyle='--')
                        ax[0].set_title('Unified Emotions (Facial + Voice)', fontsize=14, fontweight='bold')
                        ax[0].set_ylabel('Emotion Level')
                        ax[0].legend()
                        ax[0].grid(True, alpha=0.3)
                        
                        ax[1].plot(time_data, df['facial_happy'], label='Facial Happy', linewidth=2)
                        if 'voice_happy' in df.columns:
                            ax[1].plot(time_data, df['voice_happy'], label='Voice Happy', linewidth=2, linestyle='--')
                        ax[1].set_title('Happiness: Facial vs Voice', fontsize=14, fontweight='bold')
                        ax[1].set_xlabel('Time (seconds)')
                        ax[1].set_ylabel('Emotion Probability')
                        ax[1].legend()
                        ax[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(unified_filename, dpi=150, bbox_inches='tight')
                        plt.close()
                    else:
                        tracker.plot_unified_emotions(save_path=unified_filename)
                    print(f"✅ Unified analysis saved to {unified_filename}")
                
                if viz_choice in ["2", "9", "11"]:
                    plot_filename = f"{output_prefix}_facial_emotions.png"
                    bot.emotion_data = [{
                        'timestamp': row.get('timestamp'),
                        'angry': row.get('facial_angry', 0),
                        'disgust': row.get('facial_disgust', 0),
                        'fear': row.get('facial_fear', 0),
                        'happy': row.get('facial_happy', 0),
                        'sad': row.get('facial_sad', 0),
                        'surprise': row.get('facial_surprise', 0),
                        'neutral': row.get('facial_neutral', 0),
                        'arousal': row.get('facial_arousal', 0),
                        'valence': row.get('facial_valence', 0),
                        'intensity': row.get('facial_intensity', 0),
                        'excitement': row.get('facial_excitement', 0),
                        'calmness': row.get('facial_calmness', 0),
                        'positivity': row.get('facial_positivity', 0),
                        'negativity': row.get('facial_negativity', 0),
                        'quadrant': row.get('facial_quadrant', 'Unknown')
                    } for _, row in df.iterrows()]
                    bot.plot_emotions(save_path=plot_filename)
                    print(f"✅ Facial emotions saved to {plot_filename}")
                
                if viz_choice in ["3", "9", "11"]:
                    voice_filename = f"{output_prefix}_voice_features.png"
                    # Use voice bot to plot voice features
                    if user_input.lower() == "camera":
                        # For camera mode, use the live tracker's voice bot
                        live_tracker.voice_bot.plot_voice_emotions(save_path=voice_filename)
                    else:
                        # For video mode, use the tracker's voice bot
                        tracker.voice_bot.plot_voice_emotions(save_path=voice_filename)
                    print(f"✅ Voice features saved to {voice_filename}")
                
                if viz_choice in ["4", "9", "11"]:
                    heatmap_filename = f"{output_prefix}_facial_heatmap.png"
                    bot.emotion_data = [{
                        'timestamp': row.get('timestamp'),
                        'angry': row.get('facial_angry', 0),
                        'disgust': row.get('facial_disgust', 0),
                        'fear': row.get('facial_fear', 0),
                        'happy': row.get('facial_happy', 0),
                        'sad': row.get('facial_sad', 0),
                        'surprise': row.get('facial_surprise', 0),
                        'neutral': row.get('facial_neutral', 0),
                        'arousal': row.get('facial_arousal', 0),
                        'valence': row.get('facial_valence', 0),
                        'intensity': row.get('facial_intensity', 0),
                        'quadrant': row.get('facial_quadrant', 'Unknown')
                    } for _, row in df.iterrows()]
                    bot.plot_heatmap(save_path=heatmap_filename)
                    print(f"✅ Facial heatmap saved to {heatmap_filename}")
                
                if viz_choice in ["5", "9", "11"]:
                    voice_heatmap_filename = f"{output_prefix}_voice_heatmap.png"
                    # Use voice bot to plot voice heatmap
                    if user_input.lower() == "camera":
                        # For camera mode, use the live tracker's voice bot
                        live_tracker.voice_bot.plot_voice_heatmap(save_path=voice_heatmap_filename)
                    else:
                        # For video mode, use the tracker's voice bot
                        tracker.voice_bot.plot_voice_heatmap(save_path=voice_heatmap_filename)
                    print(f"✅ Voice heatmap saved to {voice_heatmap_filename}")
                
                if viz_choice in ["6", "9", "11"]:
                    movement_filename = f"{output_prefix}_movement_heatmap.png"
                    # Populate bot.emotion_data for camera mode
                    if user_input.lower() == "camera":
                        bot.emotion_data = [{
                            'timestamp': row.get('timestamp'),
                            'angry': row.get('facial_angry', 0),
                            'disgust': row.get('facial_disgust', 0),
                            'fear': row.get('facial_fear', 0),
                            'happy': row.get('facial_happy', 0),
                            'sad': row.get('facial_sad', 0),
                            'surprise': row.get('facial_surprise', 0),
                            'neutral': row.get('facial_neutral', 0),
                            'arousal': row.get('facial_arousal', 0),
                            'valence': row.get('facial_valence', 0),
                            'quadrant': row.get('facial_quadrant', 'Unknown')
                        } for _, row in df.iterrows()]
                    bot.plot_circle_movement_heatmap(save_path=movement_filename)
                    print(f"✅ Circle movement heatmap saved to {movement_filename}")
                
                if viz_choice in ["7", "9", "11"]:
                    voice_movement_filename = f"{output_prefix}_voice_movement_heatmap.png"
                    # Use voice bot to plot voice movement heatmap
                    if user_input.lower() == "camera":
                        # For camera mode, use the live tracker's voice bot
                        live_tracker.voice_bot.plot_voice_movement_heatmap(save_path=voice_movement_filename)
                    else:
                        # For video mode, use the tracker's voice bot
                        tracker.voice_bot.plot_voice_movement_heatmap(save_path=voice_movement_filename)
                    print(f"✅ Voice movement heatmap saved to {voice_movement_filename}")
                
                if viz_choice in ["8", "9", "11"]:
                    report_filename = f"{output_prefix}_report.png"
                    # Populate bot.emotion_data for camera mode
                    if user_input.lower() == "camera":
                        bot.emotion_data = [{
                            'timestamp': row.get('timestamp'),
                            'angry': row.get('facial_angry', 0),
                            'disgust': row.get('facial_disgust', 0),
                            'fear': row.get('facial_fear', 0),
                            'happy': row.get('facial_happy', 0),
                            'sad': row.get('facial_sad', 0),
                            'surprise': row.get('facial_surprise', 0),
                            'neutral': row.get('facial_neutral', 0),
                            'arousal': row.get('facial_arousal', 0),
                            'valence': row.get('facial_valence', 0),
                            'quadrant': row.get('facial_quadrant', 'Unknown')
                        } for _, row in df.iterrows()]
                    bot.generate_layperson_report(save_path=report_filename)
                    print(f"✅ Easy-to-read report saved to {report_filename}")
                
                # NEW: Comprehensive reports option (with voice data)
                if viz_choice in ["10", "11", "12"]:
                    print("\n🌟 Generating comprehensive reports with ALL data...")
                    from generate_facial_report import ComprehensiveFacialReport
                    from generate_comprehensive_voice_report import ComprehensiveVoiceReport
                    
                    # Generate facial comprehensive report
                    facial_reporter = ComprehensiveFacialReport()
                    facial_comp_filename = f"{output_prefix}_facial_comprehensive.png"
                    facial_reporter.generate_report(df, save_path=facial_comp_filename)
                    
                    # Generate voice comprehensive report
                    voice_reporter = ComprehensiveVoiceReport()
                    voice_comp_filename = f"{output_prefix}_voice_comprehensive.png"
                    voice_reporter.generate_report(df, save_path=voice_comp_filename)
                    
                    print(f"\n✅ Comprehensive reports complete!")
                    print(f"   📄 Facial: {facial_comp_filename}")
                    print(f"   📄 Voice: {voice_comp_filename}")
                
                # ULTIMATE MEGA REPORT
                if viz_choice == "12":
                    print("\n🚀 Generating ULTIMATE MEGA REPORT (24+ subplots)...")
                    from generate_ultimate_report import UltimateReportGenerator
                    
                    ultimate_generator = UltimateReportGenerator()
                    ultimate_filename = f"{output_prefix}_ULTIMATE_MEGA_REPORT.png"
                    ultimate_generator.generate_mega_report(df, save_path=ultimate_filename)
                    
                    print(f"\n✅ ULTIMATE MEGA REPORT COMPLETE!")
                    print(f"   📄 {ultimate_filename}")
                    print(f"   This single file contains EVERYTHING - all visualizations combined!")
                    
            else:
                # Facial-only visualizations (webcam mode)
                if viz_choice in ["1", "5"]:
                    plot_filename = f"{output_prefix}_emotions.png"
                    bot.plot_emotions(save_path=plot_filename)
                    print(f"✅ Technical line plot saved to {plot_filename}")
                
                if viz_choice in ["2", "5"]:
                    heatmap_filename = f"{output_prefix}_heatmap.png"
                    bot.plot_heatmap(save_path=heatmap_filename)
                    print(f"✅ Technical heatmap saved to {heatmap_filename}")
                
                if viz_choice in ["3", "5"]:
                    movement_filename = f"{output_prefix}_movement_heatmap.png"
                    bot.plot_circle_movement_heatmap(save_path=movement_filename)
                    print(f"✅ Circle movement heatmap saved to {movement_filename}")
                
                if viz_choice in ["4", "5"]:
                    report_filename = f"{output_prefix}_report.png"
                    bot.generate_layperson_report(save_path=report_filename)
                    print(f"✅ Easy-to-read report saved to {report_filename}")
                
                # Comprehensive facial report (without voice)
                if viz_choice == "6":
                    print("\n🌟 Generating comprehensive facial report with ALL data...")
                    from generate_facial_report import ComprehensiveFacialReport
                    
                    facial_reporter = ComprehensiveFacialReport()
                    facial_comp_filename = f"{output_prefix}_facial_comprehensive.png"
                    facial_reporter.generate_report(df, save_path=facial_comp_filename)
                    
                    print(f"\n✅ Comprehensive facial report complete!")
                    print(f"   📄 {facial_comp_filename}")
        else:
            # Just show the plots (not saving)
            if has_voice_data:
                if viz_choice in ["1", "9", "11"]:
                    tracker.plot_unified_emotions()
                if viz_choice in ["2", "9", "11"]:
                    bot.plot_emotions()
                if viz_choice in ["3", "9", "11"]:
                    if user_input.lower() == "camera":
                        live_tracker.voice_bot.plot_voice_emotions()
                    else:
                        tracker.voice_bot.plot_voice_emotions()
                if viz_choice in ["4", "9", "11"]:
                    bot.plot_heatmap()
                if viz_choice in ["5", "9", "11"]:
                    if user_input.lower() == "camera":
                        live_tracker.voice_bot.plot_voice_heatmap()
                    else:
                        tracker.voice_bot.plot_voice_heatmap()
                if viz_choice in ["6", "9", "11"]:
                    bot.plot_circle_movement_heatmap()
                if viz_choice in ["7", "9", "11"]:
                    if user_input.lower() == "camera":
                        live_tracker.voice_bot.plot_voice_movement_heatmap()
                    else:
                        tracker.voice_bot.plot_voice_movement_heatmap()
                if viz_choice in ["8", "9", "11"]:
                    bot.generate_layperson_report()
            else:
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

#

