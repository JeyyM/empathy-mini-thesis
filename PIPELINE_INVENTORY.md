# Video Emotion Analysis Pipeline - Complete Inventory

## 📋 Overview
This is the **starting point** of the thesis - the pipeline that processes videos to extract emotion data from facial expressions and voice, then generates comprehensive reports.

---

## 🎬 Core Detection Scripts

### 1. **`emotion_bot.py`** - Facial Emotion Detection
**Location:** `0 original/emotion_bot.py`

**What it does:**
- Detects facial emotions from video frames using FER (Facial Emotion Recognition)
- Extracts 7 basic emotions: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
- Calculates emotional dimensions using circumplex model:
  - **Arousal** - Low to high energy/activation
  - **Valence** - Negative to positive emotion
  - **Excitement** - Engagement level
  - **Positivity** - Overall positive affect
- Maps emotions to quadrant states:
  - **Excited** (High arousal + Positive valence)
  - **Stressed** (High arousal + Negative valence)
  - **Calm** (Low arousal + Positive valence)
  - **Tired** (Low arousal + Negative valence)

**Key Features:**
- Uses MTCNN for face detection
- Frame-by-frame emotion analysis
- Time-synchronized emotion tracking
- Fallback to OpenCV if FER unavailable

**Technology:**
- FER library (Facial Emotion Recognition)
- OpenCV for video processing
- MoviePy for video handling
- Optional: DeepFace for extended analysis

---

### 2. **`voice_emotion_bot.py`** - Voice Emotion Detection
**Location:** `0 original/voice_emotion_bot.py`

**What it does:**
- Analyzes audio extracted from video for voice-based emotions
- Uses **Wav2vec2 transformer ML model** (93%+ accuracy) for emotion detection
- Extracts 7 basic emotions: `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
- Calculates acoustic features:
  - **Prosodic:** Pitch, speaking rate, volume, voice tremor
  - **Spectral:** MFCCs, spectral characteristics
  - **Voice quality:** Harmonic-to-noise ratio
- Calculates emotional dimensions:
  - **Arousal** - Voice energy/activation
  - **Valence** - Positive/negative tone
  - **Dominance** - Assertiveness/confidence
  - **Intensity** - Overall emotional strength
  - **Stress** - Pressure/tension in voice

**Key Features:**
- ML-based emotion detection (Wav2vec2) as primary method
- Rule-based fallback if ML model unavailable
- Comprehensive acoustic feature extraction
- Time-synchronized with video
- Model download required: `download_emotion_model.py`

**Technology:**
- Librosa for audio analysis
- Audonnx for Wav2vec2 model
- MoviePy for audio extraction
- Scikit-learn for feature scaling

---

### 3. **`unified_emotion_tracker.py`** - Multimodal Integration
**Location:** `0 original/unified_emotion_tracker.py`

**What it does:**
- **Orchestrates** the complete analysis pipeline
- Extracts audio from video file
- Processes facial emotions (calls `emotion_bot.py`)
- Processes voice emotions (calls `voice_emotion_bot.py`)
- **Synchronizes** facial and voice data by timestamp
- Creates unified dataset with both modalities

**Process:**
1. Extract audio from video → `temp_audio.wav`
2. Analyze facial emotions → Facial DataFrame
3. Analyze voice emotions → Voice DataFrame
4. Merge by timestamp → Unified DataFrame

**Output Structure:**
- Time columns: `time_seconds`, `timestamp`
- Facial columns: `facial_angry`, `facial_disgust`, `facial_fear`, etc.
- Facial dimensions: `facial_arousal`, `facial_valence`, `facial_dominance`
- Facial states: `facial_quadrant` (Excited/Stressed/Calm/Tired)
- Voice columns: `voice_angry`, `voice_disgust`, `voice_fear`, etc.
- Voice dimensions: `voice_arousal`, `voice_valence`, `voice_dominance`, `voice_intensity`, `voice_stress`
- Voice acoustic: `voice_pitch_mean`, `voice_volume_mean`, `voice_mfcc_*`

**Technology:**
- Integrates EmotionBot + VoiceEmotionBot
- MoviePy for video/audio handling
- Pandas for data synchronization

---

### 4. **`fusion.py`** - Multimodal Fusion
**Location:** `0 original/fusion.py`

**What it does:**
- Combines facial and voice emotion data into **fused** features
- Uses **weighted averaging**: 70% facial + 30% voice
- Creates fused emotions, dimensions, and states
- Rationale: Facial expressions are more reliable than voice in controlled settings

**Fusion Formula:**
```
fused_emotion = (0.7 × facial_emotion) + (0.3 × voice_emotion)
```

**Output Columns:**
- Fused emotions: `fused_angry`, `fused_disgust`, `fused_fear`, etc.
- Fused dimensions: `fused_arousal`, `fused_valence`, `fused_dominance`, `fused_intensity`
- Fused state: `fused_quadrant` (Excited/Stressed/Calm/Tired)

**Technology:**
- Pandas for weighted averaging
- Preserves original facial and voice columns

---

## 📊 Report Generation Scripts

### 5. **`facial_reports.py`** - Facial Visualization Reports
**Location:** `0 original/facial_reports.py`

**Generates 3 comprehensive facial emotion reports:**

#### **Page 1: Facial Emotions Report** (`*_facial_emotions.png`)
- **Plot 1:** All 7 emotions over time (line graph)
- **Plot 2:** Highest overall emotion over time (line graph)
- **Plot 3:** Summary statistics table (mean, std, min, max for each emotion)
- **Plot 4:** Emotion proportion pie chart with percentages

#### **Page 2: Facial Dimensions Report** (`*_facial_dimensions.png`)
- **Plot 1:** Arousal over time (line graph)
- **Plot 2:** Valence over time (line graph)
- **Plot 3:** Arousal-Valence circumplex plot (scatter with quadrants)
- **Plot 4:** Dimensions summary statistics (mean, range for arousal/valence)

#### **Page 3: Facial States Report** (`*_facial_states.png`)
- **Plot 1:** Quadrant state over time (categorical line)
- **Plot 2:** State distribution pie chart (Excited/Stressed/Calm/Tired)
- **Plot 3:** State transition heatmap (shows state changes)
- **Plot 4:** State duration bar chart (time spent in each state)

**Color Scheme:**
- Emotions: Red (angry), Orange (surprise), Yellow (happy), Green (disgust), Blue (sad), Violet (fear), Gray (neutral)
- Dimensions: Red (intensity), Blue (valence), Green (arousal)
- States: Orange (stressed), Yellow (excited), Blue (tired), Green (calm)

---

### 6. **`voice_reports.py`** - Voice Visualization Reports
**Location:** `0 original/voice_reports.py`

**Generates 4 comprehensive voice emotion reports:**

#### **Page 1: Voice Emotions Report** (`*_voice_emotions.png`)
- **Plot 1:** All 7 emotions over time (line graph)
- **Plot 2:** Highest overall emotion over time (line graph)
- **Plot 3:** Summary statistics table (mean, std, min, max for each emotion)
- **Plot 4:** Emotion proportion pie chart with percentages

#### **Page 2: Voice Acoustic Features Report** (`*_voice_acoustic.png`)
- **Plot 1:** Pitch (mean) over time
- **Plot 2:** Volume (mean) over time
- **Plot 3:** Speaking rate over time
- **Plot 4:** Voice tremor over time
- **Plot 5:** Acoustic features correlation heatmap
- **Plot 6:** Acoustic summary statistics

#### **Page 3: Voice Dimensions Report** (`*_voice_dimensions.png`)
- **Plot 1:** Arousal over time
- **Plot 2:** Valence over time
- **Plot 3:** Intensity over time
- **Plot 4:** Stress over time
- **Plot 5:** Arousal-Valence circumplex plot
- **Plot 6:** Dimensions summary statistics

#### **Page 4: Voice States Report** (`*_voice_states.png`)
- **Plot 1:** Quadrant state over time
- **Plot 2:** State distribution pie chart
- **Plot 3:** State transition heatmap
- **Plot 4:** State duration bar chart

**Uses same color scheme as facial reports for consistency**

---

### 7. **`fusion_reports.py`** - Fusion Visualization Reports
**Location:** `0 original/fusion_reports.py`

**Generates 3 comprehensive fusion emotion reports:**

#### **Page 1: Fusion Emotions Report** (`*_fusion_emotions.png`)
- **Plot 1:** All 7 fused emotions over time
- **Plot 2:** Highest overall fused emotion over time
- **Plot 3:** Summary statistics (mean, std, min, max)
- **Plot 4:** Fused emotion proportion pie chart
- **Title includes:** "70% Facial + 30% Voice"

#### **Page 2: Fusion Dimensions Report** (`*_fusion_dimensions.png`)
- **Plot 1:** Fused arousal over time
- **Plot 2:** Fused valence over time
- **Plot 3:** Fused arousal-valence circumplex plot
- **Plot 4:** Fused dimensions summary statistics

#### **Page 3: Fusion States Report** (`*_fusion_states.png`)
- **Plot 1:** Fused quadrant state over time
- **Plot 2:** Fused state distribution pie chart
- **Plot 3:** Fused state transition heatmap
- **Plot 4:** Fused state duration bar chart

**Uses same color scheme as facial/voice reports for consistency**

---

## 🚀 Main Execution Script

### 8. **`main.py`** - Pipeline Orchestrator
**Location:** `0 original/main.py`

**What it does:**
Runs the complete analysis pipeline from start to finish.

**Execution Steps:**
1. **User Input:**
   - Asks for video filename
   - Asks for sample rate (default: 1.0 second)

2. **Step 1: Analyze Video** (Unified Emotion Tracking)
   - Calls `UnifiedEmotionTracker` with ML-enhanced voice
   - Extracts facial + voice emotions
   - Saves: `[filename]_ml_emotion_data.csv`

3. **Step 2: Generate Fusion Data**
   - Calls `MultimodalEmotionFusion` (70% facial + 30% voice)
   - Saves: `[filename]_ml_fusion.csv`

4. **Step 3: Generate Facial Reports**
   - Calls `FacialReportGenerator`
   - Generates 3 PNG files (emotions, dimensions, states)

5. **Step 4: Generate Voice Reports**
   - Calls `VoiceReportGenerator`
   - Generates 4 PNG files (emotions, acoustic, dimensions, states)

6. **Step 5: Generate Fusion Reports**
   - Calls `FusionReportGenerator`
   - Generates 3 PNG files (emotions, dimensions, states)

**Summary Output:**
```
📁 Generated Files:

📊 Data Files:
   1. [filename]_ml_emotion_data.csv (Unified facial + voice)
   2. [filename]_ml_fusion.csv (Fused multimodal data)

📸 Facial Reports (3 pages):
   1. [filename]_ml_facial_emotions.png
   2. [filename]_ml_facial_dimensions.png
   3. [filename]_ml_facial_states.png

🎤 Voice Reports (4 pages):
   1. [filename]_ml_voice_emotions.png
   2. [filename]_ml_voice_acoustic.png
   3. [filename]_ml_voice_dimensions.png
   4. [filename]_ml_voice_states.png

🔮 Fusion Reports (3 pages):
   1. [filename]_ml_fusion_emotions.png
   2. [filename]_ml_fusion_dimensions.png
   3. [filename]_ml_fusion_states.png
```

---

## 📁 Output File Structure

### **For Each Participant Video:**

#### **Data Files (CSV):**
1. `[Participant]_ml_emotion_data.csv` - Unified facial + voice emotion data
   - Columns: ~50 columns
   - Includes: time, all facial emotions/dimensions/states, all voice emotions/dimensions/states/acoustic features

2. `[Participant]_ml_fusion.csv` - Fused multimodal data
   - Columns: All facial + voice + fused columns (~75 columns)
   - Includes: Original facial/voice data + weighted fused features

#### **Visualization Files (PNG):**

**Facial (3 files):**
1. `[Participant]_ml_facial_emotions.png` - 4-panel emotion analysis
2. `[Participant]_ml_facial_dimensions.png` - 4-panel arousal/valence analysis
3. `[Participant]_ml_facial_states.png` - 4-panel quadrant state analysis

**Voice (4 files):**
1. `[Participant]_ml_voice_emotions.png` - 4-panel emotion analysis
2. `[Participant]_ml_voice_acoustic.png` - 6-panel acoustic features
3. `[Participant]_ml_voice_dimensions.png` - 6-panel dimensions analysis
4. `[Participant]_ml_voice_states.png` - 4-panel quadrant state analysis

**Fusion (3 files):**
1. `[Participant]_ml_fusion_emotions.png` - 4-panel fused emotion analysis
2. `[Participant]_ml_fusion_dimensions.png` - 4-panel fused dimensions
3. `[Participant]_ml_fusion_states.png` - 4-panel fused state analysis

---

## 🗂️ Actual Output Locations

### **Results Folder Structure:**
```
0 original/results/
├── Neutral/
│   ├── MiguelBorromeo/
│   │   ├── FinalMiguelBorromeo_ml_emotion_data.csv
│   │   ├── FinalMiguelBorromeo_ml_fusion.csv
│   │   ├── FinalMiguelBorromeo_ml_facial_emotions.png
│   │   ├── FinalMiguelBorromeo_ml_facial_dimensions.png
│   │   ├── FinalMiguelBorromeo_ml_facial_states.png
│   │   ├── FinalMiguelBorromeo_ml_voice_emotions.png
│   │   ├── FinalMiguelBorromeo_ml_voice_acoustic.png
│   │   ├── FinalMiguelBorromeo_ml_voice_dimensions.png
│   │   ├── FinalMiguelBorromeo_ml_voice_states.png
│   │   ├── FinalMiguelBorromeo_ml_fusion_emotions.png
│   │   ├── FinalMiguelBorromeo_ml_fusion_dimensions.png
│   │   └── FinalMiguelBorromeo_ml_fusion_states.png
│   ├── MiguelNg/
│   ├── RandellFabico/
│   ├── RusselGalan/
│   └── RyanSo/
├── Opposing/
│   ├── [5 participants × 12 files each]
└── Similar/
    └── [5 participants × 12 files each]
```

### **Total Output Per Participant:**
- **2 CSV files** (emotion_data + fusion)
- **10 PNG files** (3 facial + 4 voice + 3 fusion)
- **12 files total** per participant

### **Total Project Output:**
- **15 participants** × 12 files = **180 files**
- **30 CSV data files**
- **150 PNG visualization files**

---

## 📊 Data Column Summary

### **Unified Emotion Data CSV Columns (~50 columns):**

**Time:**
- `time_seconds` - Timestamp in seconds
- `timestamp` - Formatted timestamp

**Facial Emotions (7):**
- `facial_angry`, `facial_disgust`, `facial_fear`, `facial_happy`, `facial_sad`, `facial_surprise`, `facial_neutral`

**Facial Dimensions (4):**
- `facial_arousal` - Energy/activation level
- `facial_valence` - Positive/negative affect
- `facial_dominance` - Control/power
- `facial_intensity` - Overall emotional strength

**Facial States (1):**
- `facial_quadrant` - Categorical state (Excited/Stressed/Calm/Tired)

**Voice Emotions (7):**
- `voice_angry`, `voice_disgust`, `voice_fear`, `voice_happy`, `voice_sad`, `voice_surprise`, `voice_neutral`

**Voice Dimensions (5):**
- `voice_arousal` - Energy in voice
- `voice_valence` - Positive/negative tone
- `voice_dominance` - Assertiveness
- `voice_intensity` - Emotional strength
- `voice_stress` - Pressure/tension

**Voice Acoustic Features (~20):**
- `voice_pitch_mean`, `voice_pitch_std` - Pitch statistics
- `voice_volume_mean`, `voice_volume_std` - Volume statistics
- `voice_speaking_rate` - Words per minute
- `voice_mfcc_1` through `voice_mfcc_13` - Mel-frequency cepstral coefficients
- `voice_spectral_centroid`, `voice_spectral_bandwidth` - Spectral features
- `voice_hnr` - Harmonic-to-noise ratio
- `voice_tremor` - Voice tremor/shakiness

**Voice States (1):**
- `voice_quadrant` - Categorical state (Excited/Stressed/Calm/Tired)

---

### **Fusion CSV Columns (~75 columns):**

**All columns from Unified Emotion Data PLUS:**

**Fused Emotions (7):**
- `fused_angry`, `fused_disgust`, `fused_fear`, `fused_happy`, `fused_sad`, `fused_surprise`, `fused_neutral`

**Fused Dimensions (4):**
- `fused_arousal` - Combined arousal (70% facial + 30% voice)
- `fused_valence` - Combined valence
- `fused_dominance` - Combined dominance
- `fused_intensity` - Combined intensity

**Fused States (1):**
- `fused_quadrant` - Combined categorical state

---

## 🔧 Supporting Scripts

### 9. **`download_emotion_model.py`** - ML Model Downloader
**Location:** `0 original/download_emotion_model.py`

**What it does:**
- Downloads the Wav2vec2 emotion model for voice analysis
- Required for ML-enhanced voice emotion detection
- Model provides 93%+ accuracy on arousal/valence prediction

**Usage:**
```bash
python download_emotion_model.py
```

**Output:**
- `emotion_model/` folder with Wav2vec2 model files

---

### 10. **`final_install.py`** - Dependency Installer
**Location:** `0 original/final_install.py`

**What it does:**
- Installs all required Python packages
- Sets up the environment for emotion analysis
- Checks for compatible versions

**Installs:**
- OpenCV (video processing)
- FER (facial emotion recognition)
- Librosa (audio analysis)
- MoviePy (video/audio handling)
- Pandas, NumPy (data manipulation)
- Matplotlib, Seaborn (visualization)
- Audonnx (Wav2vec2 model)
- And more...

---

## 🎯 Key Findings from Pipeline

### **What Gets Measured:**

**7 Basic Emotions:**
1. Angry - Frustration, irritation
2. Disgust - Revulsion, distaste
3. Fear - Anxiety, apprehension
4. Happy - Joy, contentment
5. Sad - Sorrow, melancholy
6. Surprise - Shock, astonishment
7. Neutral - Baseline, no strong emotion

**Emotional Dimensions:**
1. **Arousal** - Low (tired) to High (energized)
2. **Valence** - Negative (unpleasant) to Positive (pleasant)
3. **Dominance** - Submissive to Dominant
4. **Intensity** - Weak to Strong emotional expression
5. **Stress** - Calm to Pressured (voice only)

**Emotional States (Circumplex):**
1. **Excited** - High arousal + Positive valence
2. **Stressed** - High arousal + Negative valence
3. **Calm** - Low arousal + Positive valence
4. **Tired** - Low arousal + Negative valence

### **What Gets Generated:**
- **180 total files** (15 participants × 12 files)
- **30 CSV datasets** with timestamped emotion data
- **150 visualization reports** showing emotion patterns
- **3 modalities analyzed:** Facial, Voice, Fusion
- **Sample rate:** 1 second intervals (typical)
- **Average video duration:** ~60-90 seconds per participant

---

## 🔄 Pipeline Workflow

```
Video Input
    ↓
┌─────────────────────────────────────┐
│ unified_emotion_tracker.py          │
│                                     │
│  1. Extract audio → temp_audio.wav  │
│  2. emotion_bot.py → Facial data    │
│  3. voice_emotion_bot.py → Voice    │
│  4. Merge by timestamp              │
└─────────────────────────────────────┘
    ↓
[Participant]_ml_emotion_data.csv
    ↓
┌─────────────────────────────────────┐
│ fusion.py                           │
│                                     │
│  Fuse: 70% facial + 30% voice       │
└─────────────────────────────────────┘
    ↓
[Participant]_ml_fusion.csv
    ↓
┌─────────────────────────────────────┐
│ Report Generators                   │
│                                     │
│  facial_reports.py → 3 PNG files    │
│  voice_reports.py → 4 PNG files     │
│  fusion_reports.py → 3 PNG files    │
└─────────────────────────────────────┘
    ↓
10 Visualization Reports (PNG)
```

---

## ✅ Summary

This pipeline is the **foundation** of the entire thesis. It:

1. ✅ **Analyzes 15 participant videos** (5 Neutral, 5 Opposing, 5 Similar)
2. ✅ **Extracts facial emotions** (7 emotions + dimensions + states)
3. ✅ **Extracts voice emotions** (7 emotions + acoustic + dimensions + states)
4. ✅ **Fuses modalities** (70% facial + 30% voice weighted fusion)
5. ✅ **Generates comprehensive reports** (10 visualizations per participant)
6. ✅ **Produces datasets** (2 CSV files per participant)
7. ✅ **Creates 180 total output files** for downstream analysis

**These outputs then feed into:**
- Summary aggregation (facial_summary_merged.csv, voice_summary_merged.csv, fusion_summary_merged.csv)
- Statistical group comparison analysis
- Prediction modeling
- Correlation analysis
- And more...

---

**End of Video Analysis Pipeline Inventory**
