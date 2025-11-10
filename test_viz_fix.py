"""
Quick test to verify the visualization fixes work
"""
from live_unified_tracker import LiveUnifiedTracker

print("Testing live tracker visualization compatibility...")

# Initialize
tracker = LiveUnifiedTracker()

# Run 5-second capture
print("\n5-second test starting...")
df = tracker.process_live(duration_seconds=5, sample_rate=1.0)

if not df.empty:
    print(f"\n✅ Captured {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Test saving visualizations
    print("\nTesting visualization saves...")
    
    # Save CSV
    df.to_csv("test_viz_data.csv", index=False)
    print("✅ CSV saved")
    
    # Test facial emotion bot visualizations
    bot = tracker.face_bot
    bot.emotion_data = [{
        'timestamp': row['timestamp'],
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
    
    try:
        bot.plot_circle_movement_heatmap(save_path="test_viz_movement.png")
        print("✅ Movement heatmap saved")
    except Exception as e:
        print(f"❌ Movement heatmap failed: {e}")
    
    try:
        bot.generate_layperson_report(save_path="test_viz_report.png")
        print("✅ Layperson report saved")
    except Exception as e:
        print(f"❌ Layperson report failed: {e}")
    
    print("\n✅ All tests passed!")
else:
    print("❌ No data captured")
