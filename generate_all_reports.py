"""
Generate ALL Comprehensive Reports
Launches both facial and voice comprehensive reports
"""
import pandas as pd
import sys
from generate_facial_report import ComprehensiveFacialReport
from generate_comprehensive_voice_report import ComprehensiveVoiceReport

def generate_all_reports(csv_file):
    """
    Generate both comprehensive reports from the same data file
    
    Args:
        csv_file: Path to CSV with emotion data
    """
    print(f"\n{'='*80}")
    print("🎯 GENERATING ALL COMPREHENSIVE REPORTS")
    print(f"{'='*80}")
    print(f"Reading data from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} samples")
        
        # Determine base filename
        base_name = csv_file.replace('.csv', '')
        
        # Check if we have facial data
        has_facial = any(col.startswith('facial_') or col in ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'] 
                        for col in df.columns)
        
        # Check if we have voice data
        has_voice = any(col.startswith('voice_') for col in df.columns)
        
        print(f"\n📊 Data contains:")
        print(f"   Facial emotions: {'✅ Yes' if has_facial else '❌ No'}")
        print(f"   Voice emotions: {'✅ Yes' if has_voice else '❌ No'}")
        
        reports_generated = 0
        
        # Generate facial report
        if has_facial:
            print(f"\n{'─'*80}")
            print("🎭 Generating Comprehensive FACIAL Emotion Report...")
            print(f"{'─'*80}")
            
            facial_reporter = ComprehensiveFacialReport()
            facial_save_path = f"{base_name}_facial_comprehensive.png"
            facial_reporter.generate_report(df, save_path=facial_save_path)
            reports_generated += 1
        else:
            print("\n⚠️  Skipping facial report (no facial data found)")
        
        # Generate voice report
        if has_voice:
            print(f"\n{'─'*80}")
            print("🎤 Generating Comprehensive VOICE Emotion Report...")
            print(f"{'─'*80}")
            
            voice_reporter = ComprehensiveVoiceReport()
            voice_save_path = f"{base_name}_voice_comprehensive.png"
            voice_reporter.generate_report(df, save_path=voice_save_path)
            reports_generated += 1
        else:
            print("\n⚠️  Skipping voice report (no voice data found)")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"✅ COMPLETE! Generated {reports_generated} comprehensive report(s)")
        print(f"{'='*80}")
        
        if has_facial:
            print(f"📄 Facial report: {base_name}_facial_comprehensive.png")
        if has_voice:
            print(f"📄 Voice report: {base_name}_voice_comprehensive.png")
        
        print(f"\n💡 These reports visualize ALL collected data points!")
        print(f"{'='*80}\n")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found: {csv_file}")
        print("\nUsage: python generate_all_reports.py [csv_file]")
        print("Example: python generate_all_reports.py webcam_live_emotion_data.csv")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # Try to find most recent CSV
        import glob
        import os
        
        csv_files = glob.glob("*emotion_data.csv")
        if csv_files:
            # Get most recently modified
            csv_file = max(csv_files, key=os.path.getmtime)
            print(f"\n💡 No file specified. Using most recent CSV: {csv_file}")
        else:
            print("\n❌ No CSV file specified and no *emotion_data.csv files found")
            print("\nUsage: python generate_all_reports.py [csv_file]")
            print("Example: python generate_all_reports.py webcam_live_emotion_data.csv")
            sys.exit(1)
    
    generate_all_reports(csv_file)
