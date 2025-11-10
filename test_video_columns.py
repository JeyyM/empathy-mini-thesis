"""Quick test to see all columns in video mode"""
from unified_emotion_tracker import UnifiedEmotionTracker

tracker = UnifiedEmotionTracker()
df = tracker.process_video_unified('happy.mp4', sample_interval=5.0)  # Large interval for faster test

print(f"\n{'='*80}")
print(f"Total columns: {len(df.columns)}")
print(f"{'='*80}")

print("\n📸 FACIAL columns:")
facial_cols = [c for c in df.columns if c.startswith('facial_')]
for col in facial_cols:
    print(f"  - {col}")

print(f"\n🎤 VOICE columns:")
voice_cols = [c for c in df.columns if c.startswith('voice_')]
for col in voice_cols:
    print(f"  - {col}")

print(f"\n⏱️  OTHER columns:")
other_cols = [c for c in df.columns if not c.startswith('facial_') and not c.startswith('voice_')]
for col in other_cols:
    print(f"  - {col}")

print(f"\n{'='*80}")
print(f"Facial: {len(facial_cols)}, Voice: {len(voice_cols)}, Other: {len(other_cols)}")
print(f"{'='*80}\n")

# Save CSV to inspect
df.to_csv("test_video_mode_columns.csv", index=False)
print("✅ Saved to test_video_mode_columns.csv for inspection")
