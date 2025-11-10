"""
Quick test of live facial + voice emotion tracking
"""
from live_unified_tracker import LiveUnifiedTracker

print("="*70)
print("TESTING LIVE UNIFIED EMOTION TRACKER")
print("="*70)
print("\nThis will capture:")
print("  📸 Facial expressions from webcam")
print("  🎤 Voice emotions from microphone")
print("\nMake sure:")
print("  - Webcam is connected and working")
print("  - Microphone is connected and working")
print("  - You're in a quiet environment")
print("\nDuring the test:")
print("  - Speak clearly (say anything)")
print("  - Make different facial expressions")
print("  - Watch the webcam window for visual feedback")
print("="*70)

input("\nPress ENTER to start 15-second test...")

# Initialize tracker
tracker = LiveUnifiedTracker()

# Run 15-second test
print("\n🎬 Starting capture... SPEAK and make expressions!")
df = tracker.process_live(duration_seconds=15, sample_rate=1.0)

if not df.empty:
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    summary = tracker.get_summary(df)
    
    print(f"\n📊 Captured {summary['total_samples']} samples over {summary['duration']:.1f} seconds")
    
    # Check what data we got
    has_facial = any(k.startswith('facial_') for k in summary.keys())
    has_voice = any(k.startswith('voice_') for k in summary.keys())
    
    print(f"\n✅ Facial data: {'YES' if has_facial else 'NO'}")
    print(f"✅ Voice data:  {'YES' if has_voice else 'NO'}")
    
    if has_facial:
        print("\n📸 Average Facial Emotions:")
        for key in ['facial_happy', 'facial_sad', 'facial_angry', 'facial_fear']:
            if key in summary:
                print(f"   {key.replace('facial_', '').title():8s}: {summary[key]:.3f}")
    
    if has_voice:
        print("\n🎤 Average Voice Emotions:")
        for key in ['voice_happy', 'voice_sad', 'voice_angry', 'voice_fear']:
            if key in summary:
                print(f"   {key.replace('voice_', '').title():8s}: {summary[key]:.3f}")
    
    if 'combined_arousal' in summary:
        print("\n🔄 Combined Metrics:")
        print(f"   Arousal: {summary['combined_arousal']:.3f}")
        print(f"   Valence: {summary['combined_valence']:.3f}")
    
    # Save results
    print("\n💾 Saving results...")
    csv_file = "live_unified_test.csv"
    df.to_csv(csv_file, index=False)
    print(f"   ✅ Data saved to {csv_file}")
    print(f"   📋 Columns: {len(df.columns)}")
    print(f"   📊 Rows: {len(df)}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    
else:
    print("\n❌ No data captured - please check webcam and microphone")
