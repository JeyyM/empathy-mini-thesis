import pandas as pd
import numpy as np

df = pd.read_csv('angry_emotion_data.csv')

print("=" * 80)
print("ANGRY VIDEO - VOICE ACOUSTIC FEATURES ANALYSIS")
print("=" * 80)

print("\n--- DIMENSIONS (Should show high arousal + negative valence for angry) ---")
print(f"Arousal:   Mean={df['voice_arousal'].mean():7.3f}  Range=[{df['voice_arousal'].min():7.3f}, {df['voice_arousal'].max():7.3f}]")
print(f"Valence:   Mean={df['voice_valence'].mean():7.3f}  Range=[{df['voice_valence'].min():7.3f}, {df['voice_valence'].max():7.3f}]")
print(f"Intensity: Mean={df['voice_intensity'].mean():7.3f}  Range=[{df['voice_intensity'].min():7.3f}, {df['voice_intensity'].max():7.3f}]")
print(f"Stress:    Mean={df['voice_stress'].mean():7.3f}  Range=[{df['voice_stress'].min():7.3f}, {df['voice_stress'].max():7.3f}]")

print("\n--- RAW ACOUSTIC FEATURES ---")
print(f"Pitch Mean:       {df['voice_pitch_mean'].mean():7.1f} Hz")
print(f"Pitch Variation:  {df['voice_pitch_variation'].mean():7.3f}")
print(f"Volume Mean:      {df['voice_volume_mean'].mean():7.3f}")
print(f"Volume Std:       {df['voice_volume_std'].mean():7.3f}")
print(f"Harmonic Ratio:   {df['voice_harmonic_ratio'].mean():7.3f}")
print(f"Silence Ratio:    {df['voice_silence_ratio'].mean():7.3f}")
print(f"Voice Tremor:     {df['voice_tremor'].mean():7.1f}")
print(f"Speech Rate:      {df['voice_speech_rate'].mean():7.1f}")

print("\n--- EMOTION SCORES (Current - WRONG) ---")
emotions = ['voice_angry', 'voice_disgust', 'voice_fear', 'voice_happy', 
            'voice_sad', 'voice_surprise', 'voice_neutral']
for emotion in emotions:
    if emotion in df.columns:
        avg = df[emotion].mean()
        print(f"{emotion.replace('voice_', '').capitalize():10s}: {avg:.3f} ({avg*100:.1f}%)")

print("\n--- PROBLEM DIAGNOSIS ---")
print("\n1. VALENCE IS TOO POSITIVE!")
print(f"   Current: {df['voice_valence'].mean():.3f} (should be negative for angry)")
print(f"   The formula is giving positive valence instead of negative")

print("\n2. AROUSAL IS TOO LOW!")
print(f"   Current: {df['voice_arousal'].mean():.3f} (should be high for angry)")
print(f"   Angry voices should have arousal close to 1.0")

print("\n3. PITCH VARIATION IS MODERATE")
print(f"   Current: {df['voice_pitch_variation'].mean():.3f}")
print(f"   This is actually good for angry detection")

print("\n4. VOLUME IS LOW")
print(f"   Current: {df['voice_volume_mean'].mean():.3f}")
print(f"   This is VERY low - might be the issue!")

print("\n5. HARMONIC RATIO IS MODERATE")
print(f"   Current: {df['voice_harmonic_ratio'].mean():.3f}")
print(f"   Angry voices should have lower harmonic ratio (harsh/distorted)")

print("\n--- RECOMMENDED FIXES ---")
print("1. Fix VALENCE formula - spectral features are making it too positive")
print("2. Boost AROUSAL sensitivity - pitch variation should contribute more")
print("3. Increase ANGRY emotion weight - current 1.25x multiplier is too low")
print("4. Lower HAPPY emotion - positive valence is incorrectly high")
print("5. Volume normalization might be wrong - 0.04 volume is being normalized incorrectly")

print("\n" + "=" * 80)
