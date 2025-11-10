"""
Quick test for comprehensive reports
Tests both facial and voice comprehensive report generation
"""
import pandas as pd
import numpy as np
from generate_facial_report import ComprehensiveFacialReport
from generate_comprehensive_voice_report import ComprehensiveVoiceReport

def create_test_data(duration=10, sample_rate=2):
    """Create synthetic test data with all features"""
    samples = int(duration * sample_rate)
    time = np.linspace(0, duration, samples)
    
    # Create data dictionary
    data = {
        'timestamp': time,
        'time_seconds': time,
    }
    
    # Facial emotions (7 basic)
    emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
    for i, emotion in enumerate(emotions):
        # Create varied patterns
        data[f'facial_{emotion}'] = 0.1 + 0.3 * np.sin(2 * np.pi * time / duration + i) ** 2
    
    # Facial dimensions
    data['facial_arousal'] = 0.5 * np.sin(2 * np.pi * time / duration)
    data['facial_valence'] = 0.3 * np.cos(2 * np.pi * time / duration)
    data['facial_intensity'] = 0.4 + 0.2 * np.sin(4 * np.pi * time / duration)
    data['facial_excitement'] = data['facial_arousal'] * 0.6
    data['facial_calmness'] = -data['facial_arousal'] * 0.6
    data['facial_positivity'] = data['facial_valence'] * 0.7
    data['facial_negativity'] = -data['facial_valence'] * 0.7
    
    # Quadrant
    quadrants = []
    for arousal, valence in zip(data['facial_arousal'], data['facial_valence']):
        if arousal > 0 and valence > 0:
            quadrants.append('Excited')
        elif arousal > 0 and valence < 0:
            quadrants.append('Agitated')
        elif arousal < 0 and valence > 0:
            quadrants.append('Calm')
        else:
            quadrants.append('Depressed')
    data['facial_quadrant'] = quadrants
    
    # Voice emotions (4 basic)
    voice_emotions = ['happy', 'sad', 'angry', 'neutral']
    for i, emotion in enumerate(voice_emotions):
        data[f'voice_{emotion}'] = 0.15 + 0.25 * np.sin(2 * np.pi * time / duration + i * 1.5) ** 2
    
    # Voice dimensions
    data['voice_arousal'] = 0.4 * np.sin(2 * np.pi * time / duration + 0.5)
    data['voice_valence'] = 0.35 * np.cos(2 * np.pi * time / duration + 0.3)
    
    # Voice acoustic features
    data['voice_pitch_mean'] = 150 + 50 * np.sin(2 * np.pi * time / duration)
    data['voice_pitch_std'] = 10 + 5 * np.abs(np.sin(4 * np.pi * time / duration))
    data['voice_volume_mean'] = -20 + 5 * np.sin(2 * np.pi * time / duration)
    data['voice_volume_std'] = 2 + 1 * np.abs(np.cos(3 * np.pi * time / duration))
    
    # Spectral features
    data['voice_spectral_centroid'] = 2000 + 500 * np.sin(2 * np.pi * time / duration)
    data['voice_spectral_rolloff'] = 4000 + 1000 * np.sin(2 * np.pi * time / duration + 0.5)
    data['voice_zero_crossing_rate'] = 0.1 + 0.05 * np.abs(np.sin(3 * np.pi * time / duration))
    
    # MFCCs (13 coefficients)
    for i in range(1, 14):
        data[f'voice_mfcc_{i}'] = np.random.randn(samples) * (20 - i)
    
    # Prosody features
    data['voice_speaking_rate'] = 3 + 0.5 * np.sin(2 * np.pi * time / duration)
    data['voice_tempo'] = 120 + 20 * np.sin(2 * np.pi * time / duration)
    data['voice_pitch_range'] = 50 + 20 * np.abs(np.sin(2 * np.pi * time / duration))
    data['voice_jitter'] = 0.01 + 0.005 * np.random.rand(samples)
    data['voice_shimmer'] = 0.05 + 0.02 * np.random.rand(samples)
    
    return pd.DataFrame(data)

def test_comprehensive_reports():
    """Test comprehensive report generation"""
    print("\n" + "="*80)
    print("🧪 TESTING COMPREHENSIVE REPORTS")
    print("="*80)
    
    # Create test data
    print("\n📊 Creating synthetic test data...")
    df = create_test_data(duration=30, sample_rate=2)
    print(f"✅ Created {len(df)} samples with {len(df.columns)} features")
    
    # Save test data
    test_csv = "test_comprehensive_data.csv"
    df.to_csv(test_csv, index=False)
    print(f"✅ Saved test data to: {test_csv}")
    
    # Test facial report
    print("\n" + "─"*80)
    print("🎭 Testing FACIAL Comprehensive Report...")
    print("─"*80)
    try:
        facial_reporter = ComprehensiveFacialReport()
        facial_reporter.generate_report(df, save_path="test_facial_comprehensive.png")
        print("✅ Facial report generated successfully!")
    except Exception as e:
        print(f"❌ Facial report failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test voice report
    print("\n" + "─"*80)
    print("🎤 Testing VOICE Comprehensive Report...")
    print("─"*80)
    try:
        voice_reporter = ComprehensiveVoiceReport()
        voice_reporter.generate_report(df, save_path="test_voice_comprehensive.png")
        print("✅ Voice report generated successfully!")
    except Exception as e:
        print(f"❌ Voice report failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with generate_all_reports
    print("\n" + "─"*80)
    print("🌟 Testing generate_all_reports.py...")
    print("─"*80)
    try:
        from generate_all_reports import generate_all_reports
        generate_all_reports(test_csv)
        print("✅ All reports generated successfully!")
    except Exception as e:
        print(f"❌ All reports failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE REPORT TESTING COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - test_comprehensive_data.csv")
    print("  - test_facial_comprehensive.png")
    print("  - test_voice_comprehensive.png")
    print("  - test_comprehensive_data_facial_comprehensive.png")
    print("  - test_comprehensive_data_voice_comprehensive.png")
    print("\n💡 Open the PNG files to verify the reports look correct!")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_comprehensive_reports()
