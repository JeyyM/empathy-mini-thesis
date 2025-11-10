"""
Test all visualization options with happy.mp4
"""
print("="*80)
print("TESTING ALL VISUALIZATION OPTIONS")
print("="*80)

print("\n📊 WITH VOICE DATA (Video Mode) - 9 Options:")
print("1. Unified analysis (facial + voice combined)")
print("2. Facial emotions line plot")
print("3. Voice features only")
print("4. Facial emotion heatmap")
print("5. Circle movement heatmap")
print("6. Easy-to-read report")
print("7. All standard visualizations (1-6)")
print("8. Comprehensive reports (facial + voice separate)")
print("9. EVERYTHING (all standard + comprehensive)")

print("\n" + "="*80)
print("Testing Option 9 - EVERYTHING")
print("="*80)

import subprocess
import os

test_input = "happy.mp4\n10.0\ny\n9\n"

process = subprocess.Popen(
    ["python", "main.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',
    errors='replace'
)

output, _ = process.communicate(input=test_input, timeout=300)

# Print last 40 lines
lines = output.split('\n')
print('\n'.join(lines[-40:]))

print("\n" + "="*80)
print("EXPECTED FILES FROM OPTION 9")
print("="*80)

expected_files = {
    "happy_emotion_data.csv": "Raw data",
    "happy_unified_emotions.png": "1. Unified analysis",
    "happy_facial_emotions.png": "2. Facial line plot",
    "happy_voice_features.png": "3. Voice features",
    "happy_facial_heatmap.png": "4. Facial heatmap",
    "happy_movement_heatmap.png": "5. Circle movement",
    "happy_report.png": "6. Easy-to-read report",
    "happy_facial_comprehensive.png": "8. Facial comprehensive",
    "happy_voice_comprehensive.png": "8. Voice comprehensive"
}

print(f"\nChecking for {len(expected_files)} expected files:")
print("-"*80)

found = 0
for filename, description in expected_files.items():
    if os.path.exists(filename):
        size = os.path.getsize(filename) / 1024
        print(f"✅ {filename:40s} ({size:6.1f} KB) - {description}")
        found += 1
    else:
        print(f"❌ {filename:40s} - {description}")

print("-"*80)
print(f"Found {found}/{len(expected_files)} files")

if found == len(expected_files):
    print("\n🎉 SUCCESS! All files generated correctly!")
else:
    print(f"\n⚠️  {len(expected_files) - found} file(s) missing")

print("="*80)
