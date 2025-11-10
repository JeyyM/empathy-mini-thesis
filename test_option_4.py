"""
Quick test for Option 4 - Facial Heatmap
Tests the fix for arousal/valence KeyError
"""

import subprocess
import os
import time

print("="*80)
print("QUICK TEST - Option 4: Facial Heatmap")
print("="*80)

# Delete old file if exists
if os.path.exists("happy_facial_heatmap.png"):
    os.remove("happy_facial_heatmap.png")
    print("[CLEANUP] Removed old happy_facial_heatmap.png")

# Input for main.py: happy.mp4, 10 second intervals, save=yes, option=4
test_input = "happy.mp4\n10\ny\n4\n"

print("\n[TEST] Running: python main.py")
print("[INPUT] Video: happy.mp4")
print("[INPUT] Sample interval: 10 seconds")
print("[INPUT] Save: yes")
print("[INPUT] Visualization choice: 4 (Facial heatmap)")
print("\nProcessing... (this may take 2-3 minutes)\n")

try:
    process = subprocess.Popen(
        ["python", "main.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    output, errors = process.communicate(input=test_input, timeout=300)
    
    print("\n" + "="*80)
    print("STDOUT:")
    print("="*80)
    print(output)
    
    if errors:
        print("\n" + "="*80)
        print("STDERR:")
        print("="*80)
        print(errors)
    
    # Check if file was created
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    if os.path.exists("happy_facial_heatmap.png"):
        size_mb = os.path.getsize("happy_facial_heatmap.png") / (1024 * 1024)
        print(f"✅ SUCCESS: happy_facial_heatmap.png created ({size_mb:.2f} MB)")
    else:
        print("❌ FAILED: happy_facial_heatmap.png not found")
        
except subprocess.TimeoutExpired:
    print("\n⚠️ Test timed out after 300 seconds")
    process.kill()
except Exception as e:
    print(f"\n❌ Error: {e}")

print("="*80)
