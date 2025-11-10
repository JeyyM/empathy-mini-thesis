# Live Unified Emotion Tracker - Integration Complete! 🎉

## What's New

You now have **real-time facial + voice emotion tracking** built into your main system!

## How It Works

### 📸 **Webcam Mode** (NEW!)
- **Simultaneously captures:**
  - Facial expressions from webcam (FER emotion detection)
  - Voice emotions from microphone (pitch, MFCCs, prosody)
- **Real-time processing** with live feedback
- **Synchronized data** combining both modalities

### 🎬 **Video Mode** (Existing)
- Analyzes pre-recorded videos
- Extracts audio automatically
- Processes facial + voice emotions
- Generates unified reports

## Usage

### Run the System:
```bash
python main.py
```

### Choose Your Mode:

**Option 1: Live Camera (Facial + Voice)**
```
Input: camera
Duration: 30 seconds (or custom)
Sample rate: 1.0 seconds (or custom)
```
- Webcam window shows your face
- Microphone records your voice
- Press 'q' to stop early
- Get synchronized facial + voice emotion data

**Option 2: Video File (Facial + Voice)**
```
Input: happy.mp4 (or any video filename)
Sample rate: 1.0 seconds (or custom)
```
- Analyzes video file
- Extracts and analyzes audio
- Generates comprehensive reports

## Output Data

### Webcam Mode CSV Columns:
- `time_seconds` - Timestamp
- **Facial emotions:** `facial_happy`, `facial_sad`, `facial_angry`, `facial_fear`, `facial_disgust`, `facial_surprise`, `facial_neutral`
- **Facial dimensions:** `facial_arousal`, `facial_valence`, `facial_quadrant`
- **Voice emotions:** `voice_happy`, `voice_sad`, `voice_angry`, `voice_fear`, etc.
- **Voice features:** `pitch_mean`, `volume_mean`, `speech_rate`
- **Voice dimensions:** `voice_arousal`, `voice_valence`, `voice_intensity`
- **Combined metrics:** `combined_arousal`, `combined_valence`

### Video Mode CSV Columns:
- All of the above PLUS:
- More comprehensive voice features
- Additional spectral features
- Enhanced combined metrics

## Visualizations

### For Webcam (Live) Mode:
1. **Unified emotions** - Facial vs voice comparison over time
2. **Facial emotions** - Technical line plots
3. **Circle movement heatmap** - Emotional state visualization
4. **Easy-to-read report** - For everyone (recommended)

### For Video Mode:
1. **Unified emotions** - Complete facial + voice analysis
2. **Facial emotions** - Separate facial tracking
3. **Voice features** - Pitch, volume, speech rate
4. **Circle movement heatmap**
5. **Easy-to-read report**

## Technical Details

### Requirements:
- Python 3.13
- OpenCV (webcam/video)
- PyAudio (microphone)
- FER (facial emotion recognition)
- Librosa (voice analysis)

### Architecture:
```
main.py
├── LiveUnifiedTracker (camera mode)
│   ├── EmotionBot (facial)
│   ├── VoiceEmotionBot (voice)
│   └── Background audio recording thread
│
└── UnifiedEmotionTracker (video mode)
    ├── EmotionBot (facial)
    ├── VoiceEmotionBot (voice)
    └── Audio extraction (moviepy)
```

### Live Mode Process:
1. Initialize webcam + microphone
2. Start audio recording in background thread
3. Process facial emotions at sample rate
4. Save audio to file when done
5. Analyze voice emotions from recording
6. Synchronize facial + voice data by timestamp
7. Generate unified dataset

## Example Use Cases

### Research Applications:
- **Constructive listening studies** - Track defensiveness, openness
- **Multimodal emotion analysis** - Compare facial vs vocal expression
- **Real-time empathy measurement** - Live emotional state tracking
- **ChatGPT dialogue studies** - Measure engagement during conversations

### Quick Tests:
```bash
# 10-second camera test
python test_live_system.py

# Process existing video
python test_unified_system.py
```

## Files Created

### New Files:
- `live_unified_tracker.py` - Real-time facial + voice tracking
- `live_requirements.txt` - PyAudio dependency
- `test_live_system.py` - Quick test script

### Modified Files:
- `main.py` - Integrated live + video modes
- Now handles both webcam (live) and video (file) seamlessly

### Output Files (examples):
- `webcam_live_emotion_data.csv` - Live capture data
- `webcam_live_unified_emotions.png` - Visualization
- `happy_unified_emotions.png` - Video analysis visualization

## Next Steps

### Ready to Use:
✅ Run `python main.py` and select `camera` for live mode
✅ Run `python main.py` and enter a video filename for video mode
✅ System automatically handles facial + voice for both!

### Tips:
- **For best results:**
  - Good lighting for facial detection
  - Clear audio input for voice detection
  - Speak naturally and make expressions
  - Try different emotions to see the system work

- **Sample rates:**
  - `0.5` seconds = very frequent sampling (more data)
  - `1.0` seconds = standard (recommended)
  - `2.0` seconds = less frequent (faster processing)

## Summary

🎯 **Mission Accomplished!**
- ✅ Live webcam + microphone integration complete
- ✅ Facial + voice emotions captured simultaneously
- ✅ Synchronized data generation
- ✅ Works seamlessly with existing video mode
- ✅ One unified interface for both modes

Your emotion tracking system now captures the complete picture - both what people **show** on their face and what they **express** in their voice, in real-time! 🚀
