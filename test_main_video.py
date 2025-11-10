"""
Simple test of main.py with happy.mp4 to generate comprehensive reports
"""
import subprocess
import os
import time

print("="*80)
print("TESTING VIDEO MODE WITH COMPREHENSIVE REPORTS")
print("="*80)

# Prepare input
test_input = "happy.mp4\n10.0\ny\n7\n"

# Run main.py
print("\nRunning: python main.py")
print("Input: happy.mp4, sample_rate=10.0, save=y, viz_choice=7")
print("-"*80)

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

# Print last 50 lines of output
lines = output.split('\n')
print('\n'.join(lines[-50:]))

print("\n" + "="*80)
print("CHECKING GENERATED FILES")
print("="*80)

expected_files = [
    "happy_emotion_data.csv",
    "happy_facial_comprehensive.png",
    "happy_voice_comprehensive.png"
]

for filename in expected_files:
    if os.path.exists(filename):
        size = os.path.getsize(filename) / 1024  # KB
        print(f"✅ {filename} ({size:.1f} KB)")
    else:
        print(f"❌ {filename} - NOT FOUND")

print("="*80)
