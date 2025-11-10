"""
Quick test for new voice heatmap feature (Option 5)
"""

import subprocess
import os

print("="*80)
print("TESTING NEW FEATURE: Voice Heatmap (Option 5)")
print("="*80)

# Input for main.py: happy.mp4, 10 second intervals, save=yes, option=5
test_input = "happy.mp4\n10\ny\n5\n"

print("\n[TEST] Running: python main.py")
print("[INPUT] Video: happy.mp4")
print("[INPUT] Sample interval: 10 seconds")
print("[INPUT] Save: yes")
print("[INPUT] Visualization choice: 5 (Voice heatmap)")
print("\nProcessing... (this may take 2-3 minutes)\n")

try:
    process = subprocess.Popen(
        ["python", "main.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    output, _ = process.communicate(input=test_input, timeout=300)
    
    print("\n" + "="*80)
    print("OUTPUT:")
    print("="*80)
    print(output)
    
    # Check if file was created
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    if os.path.exists("happy_voice_heatmap.png"):
        size_mb = os.path.getsize("happy_voice_heatmap.png") / (1024 * 1024)
        print(f"✅ SUCCESS: happy_voice_heatmap.png created ({size_mb:.2f} MB)")
    else:
        print("❌ FAILED: happy_voice_heatmap.png not found")
        
except subprocess.TimeoutExpired:
    print("\n⚠️ Test timed out after 300 seconds")
    process.kill()
except Exception as e:
    print(f"\n❌ Error: {e}")

print("="*80)
