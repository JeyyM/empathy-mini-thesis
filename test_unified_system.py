"""
Quick test of the unified facial + voice emotion system
"""
from unified_emotion_tracker import UnifiedEmotionTracker

print("="*70)
print("TESTING UNIFIED EMOTION TRACKER (Facial + Voice)")
print("="*70)

# Initialize tracker
tracker = UnifiedEmotionTracker()

# Test with happy video
video_path = "happy.mp4"
print(f"\n🎬 Processing video: {video_path}")
print("This will analyze BOTH facial expressions AND voice emotions\n")

# Process the video
df = tracker.process_video_unified(video_path, sample_interval=1.0)

if not df.empty:
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    # Check what columns we have
    has_facial = 'facial_happy' in df.columns
    has_voice = 'voice_happy' in df.columns
    
    print(f"\n📊 Captured {len(df)} synchronized samples")
    print(f"   Facial data: {'✅ YES' if has_facial else '❌ NO'}")
    print(f"   Voice data:  {'✅ YES' if has_voice else '❌ NO'}")
    
    if has_facial:
        print("\n📸 FACIAL Emotions (top 3):")
        facial_emotions = {
            'Happy': df['facial_happy'].mean(),
            'Neutral': df['facial_neutral'].mean(),
            'Sad': df['facial_sad'].mean(),
            'Angry': df['facial_angry'].mean()
        }
        sorted_facial = sorted(facial_emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, value in sorted_facial[:3]:
            print(f"   {emotion:>8}: {value:.3f}")
    
    if has_voice:
        print("\n🎤 VOICE Emotions (top 3):")
        voice_emotions = {
            'Happy': df['voice_happy'].mean(),
            'Neutral': df['voice_neutral'].mean(),
            'Sad': df['voice_sad'].mean(),
            'Angry': df['voice_angry'].mean()
        }
        sorted_voice = sorted(voice_emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, value in sorted_voice[:3]:
            print(f"   {emotion:>8}: {value:.3f}")
    
    if 'combined_arousal' in df.columns:
        print("\n🔄 COMBINED Metrics:")
        print(f"   Arousal:   {df['combined_arousal'].mean():>6.3f}")
        print(f"   Valence:   {df['combined_valence'].mean():>6.3f}")
        print(f"   Intensity: {df['combined_intensity'].mean():>6.3f}")
    
    # Save results
    print("\n💾 Saving results...")
    csv_file = "happy_unified_test.csv"
    df.to_csv(csv_file, index=False)
    print(f"   ✅ Data saved to {csv_file}")
    
    viz_file = "happy_unified_test.png"
    tracker.plot_unified_emotions(save_path=viz_file)
    print(f"   ✅ Visualization saved to {viz_file}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
else:
    print("❌ No data was captured")
