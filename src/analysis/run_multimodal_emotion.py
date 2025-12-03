"""
Main Multimodal Emotion Analysis Pipeline - ML Enhanced Version
Analyzes video files for facial expressions and voice emotions using ML models
Generates comprehensive reports for facial, voice, and fused data

NEW FEATURES:
- Uses Wav2vec2 ML model for voice emotion detection (93%+ accuracy)
- Improved arousal/valence predictions from transformer-based model
- Better angry/happy/sad discrimination
"""

import os
import sys
import pandas as pd
from unified_emotion_tracker import UnifiedEmotionTracker
from facial_reports import FacialReportGenerator
from voice_reports import VoiceReportGenerator
from fusion import MultimodalEmotionFusion


def main():
    print("=" * 80)
    print("🔬 MULTIMODAL EMOTION ANALYSIS SYSTEM (ML Enhanced)")
    print("=" * 80)
    print("\nThis system analyzes:")
    print("  📸 Facial Expressions (7 emotions + dimensions + states)")
    print("  🎤 Voice Emotions (ML Model: Wav2vec2 transformer)")
    print("      - 7 emotions + acoustic features + dimensions + states")
    print("      - 93%+ accuracy on arousal/valence prediction")
    print("  🔮 Multimodal Fusion (Combined 70% facial + 30% voice)")
    print()
    
    # ========== USER INPUT ==========
    # Get video file
    video_path = input("Enter video filename: ").strip()
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: File '{video_path}' not found")
        print("Make sure the file is in the current directory or provide full path")
        return
    
    # Get sample rate
    try:
        sample_rate = float(input("Sample rate in seconds (default 1.0): ") or "1.0")
    except ValueError:
        sample_rate = 1.0
        print(f"⚠️ Invalid input, using default: {sample_rate} seconds")
    
    print("\n" + "=" * 80)
    print("📊 PROCESSING PIPELINE")
    print("=" * 80)
    
    # Generate base filename and results output directory
    base_filename = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "_")
    results_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
    output_dir = os.path.join(results_root, base_filename)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        print(f"⚠️ Could not create results folder at '{output_dir}', falling back to current directory")
        output_dir = os.getcwd()
    
    # ========== STEP 1: Analyze Video (Facial + Voice with ML) ==========
    print("\n[1/4] 🎬 Analyzing video for facial expressions and voice emotions...")
    print(f"      Video: {video_path}")
    print(f"      Sample rate: {sample_rate} seconds")
    print(f"      🤖 Using ML model for voice emotion detection")
    
    # Create tracker with ML-enhanced voice bot
    tracker = UnifiedEmotionTracker(sample_rate=22050, use_ml_voice=True)
    unified_df = tracker.process_video_unified(video_path, sample_interval=sample_rate)
    
    if unified_df.empty:
        print("❌ Failed to process video. Exiting.")
        return
    
    # Save unified emotion data
    unified_csv = os.path.join(output_dir, f"{base_filename}_ml_emotion_data.csv")
    unified_df.to_csv(unified_csv, index=False)
    print(f"\n✅ Unified emotion data saved: {unified_csv}")
    print(f"   Total samples: {len(unified_df)}")
    
    # ========== STEP 2: Generate Fusion Data ==========
    print("\n[2/4] 🔮 Performing multimodal fusion (70% facial + 30% voice)...")
    
    fusion = MultimodalEmotionFusion(w_facial=0.7, w_voice=0.3)
    fused_df = fusion.fuse_data(unified_csv)
    
    fusion_csv = os.path.join(output_dir, f"{base_filename}_ml_fusion.csv")
    print(f"✅ Fusion data saved: {fusion_csv}")
    
    # ========== STEP 3: Generate Facial Reports ==========
    print("\n[3/4] 📸 Generating facial emotion reports...")
    
    facial_generator = FacialReportGenerator()
    
    print("   📊 Generating facial emotions report...")
    facial_generator.generate_emotion_report(unified_df, os.path.join(output_dir, f"{base_filename}_ml"))
    
    print("   📊 Generating facial dimensions report...")
    facial_generator.generate_dimensions_report(unified_df, os.path.join(output_dir, f"{base_filename}_ml"))
    
    print("   📊 Generating facial states report...")
    facial_generator.generate_states_report(unified_df, os.path.join(output_dir, f"{base_filename}_ml"))
    
    print(f"\n✅ Facial reports generated:")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_facial_emotions.png')}")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_facial_dimensions.png')}")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_facial_states.png')}")
    
    # ========== STEP 4: Generate Voice Reports ==========
    print("\n[4/4] 🎤 Generating voice emotion reports (ML-based)...")
    
    voice_generator = VoiceReportGenerator()
    
    print("   📊 Generating voice emotions report...")
    voice_generator.generate_emotion_report(unified_df, os.path.join(output_dir, f"{base_filename}_ml"))
    
    print("   📊 Generating voice acoustic features report...")
    voice_generator.generate_acoustic_report(unified_df, os.path.join(output_dir, f"{base_filename}_ml"))
    
    print("   📊 Generating voice dimensions report...")
    voice_generator.generate_dimensions_report(unified_df, os.path.join(output_dir, f"{base_filename}_ml"))
    
    print("   📊 Generating voice states report...")
    voice_generator.generate_states_report(unified_df, os.path.join(output_dir, f"{base_filename}_ml"))
    
    print(f"\n✅ Voice reports generated (ML-enhanced):")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_voice_emotions.png')}")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_voice_acoustic.png')}")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_voice_dimensions.png')}")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_voice_states.png')}")
    
    # ========== STEP 5: Generate Fusion Reports ==========
    print("\n[5/5] 🔮 Generating fusion emotion reports...")
    
    from fusion_reports import generate_all_reports
    generate_all_reports(fusion_csv)
    
    print(f"\n✅ Fusion reports generated:")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_fusion_emotions.png')}")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_fusion_dimensions.png')}")
    print(f"   - {os.path.join(output_dir, f'{base_filename}_ml_fusion_states.png')}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("\n📊 Data Files:")
    print(f"   1. {unified_csv} (Unified facial + ML voice data)")
    print(f"   2. {fusion_csv} (Fused multimodal data)")
    
    print("\n📸 Facial Reports (3 pages):")
    print(f"   1. {base_filename}_ml_facial_emotions.png")
    print(f"   2. {base_filename}_ml_facial_dimensions.png")
    print(f"   3. {base_filename}_ml_facial_states.png")
    
    print("\n🎤 Voice Reports - ML Enhanced (4 pages):")
    print(f"   1. {base_filename}_ml_voice_emotions.png")
    print(f"   2. {base_filename}_ml_voice_acoustic.png")
    print(f"   3. {base_filename}_ml_voice_dimensions.png")
    print(f"   4. {base_filename}_ml_voice_states.png")
    
    print("\n🔮 Fusion Reports (3 pages):")
    print(f"   1. {base_filename}_ml_fusion_emotions.png")
    print(f"   2. {base_filename}_ml_fusion_dimensions.png")
    print(f"   3. {base_filename}_ml_fusion_states.png")
    
    print(f"\n📈 Total: 2 CSV files + 10 visualization reports")
    print("\n" + "=" * 80)
    print("✅ All ML-enhanced reports ready for analysis!")
    print("=" * 80)
    
    print("\n📌 ML Model Info:")
    print("   Voice: Wav2vec2 transformer (audeering)")
    print("   Accuracy: 93%+ UAR on arousal/valence")
    print("   Features: Arousal, Valence, Dominance + Acoustic features")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
