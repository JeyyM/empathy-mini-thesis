# 📊 Complete Data Collection Reference

## All Data Points Captured by the Emotion Tracking System

This document maps out **every single piece of data** collected by the multimodal emotion tracking system.

---

## 🎭 FACIAL EMOTION DATA (from camera/video frames)

### Basic Emotions (7 categories)
Detected by FER (Facial Emotion Recognition) model using deep learning on facial features.

| Data Point | Type | Range | Description |
|------------|------|-------|-------------|
| `facial_happy` | float | 0.0 - 1.0 | Probability of happiness/joy |
| `facial_sad` | float | 0.0 - 1.0 | Probability of sadness |
| `facial_angry` | float | 0.0 - 1.0 | Probability of anger |
| `facial_fear` | float | 0.0 - 1.0 | Probability of fear/anxiety |
| `facial_surprise` | float | 0.0 - 1.0 | Probability of surprise |
| `facial_disgust` | float | 0.0 - 1.0 | Probability of disgust |
| `facial_neutral` | float | 0.0 - 1.0 | Probability of neutral expression |

### Dimensional Emotions
Derived from basic emotions using psychological research mappings.

| Data Point | Type | Range | Description |
|------------|------|-------|-------------|
| `facial_arousal` | float | -1.0 to 1.0 | Energy level (calm ↔ excited) |
| `facial_valence` | float | -1.0 to 1.0 | Mood quality (negative ↔ positive) |
| `facial_intensity` | float | 0.0 - 1.0 | Overall emotional intensity |
| `facial_excitement` | float | -1.0 to 1.0 | High arousal, positive valence |
| `facial_calmness` | float | -1.0 to 1.0 | Low arousal, positive valence |
| `facial_positivity` | float | -1.0 to 1.0 | Positive emotional tendency |
| `facial_negativity` | float | -1.0 to 1.0 | Negative emotional tendency |

### Categorical Labels

| Data Point | Type | Values | Description |
|------------|------|--------|-------------|
| `facial_quadrant` | string | Excited, Agitated, Calm, Depressed | Emotional state quadrant based on arousal-valence |
| `dominant_emotion` | string | happy, sad, angry, etc. | The highest-scoring emotion |

---

## 🎤 VOICE EMOTION DATA (from audio/microphone)

### Voice Emotions (4 categories)
Detected from acoustic features using machine learning on prosody patterns.

| Data Point | Type | Range | Description |
|------------|------|-------|-------------|
| `voice_happy` | float | 0.0 - 1.0 | Probability of happy voice |
| `voice_sad` | float | 0.0 - 1.0 | Probability of sad voice |
| `voice_angry` | float | 0.0 - 1.0 | Probability of angry voice |
| `voice_neutral` | float | 0.0 - 1.0 | Probability of neutral voice |

### Voice Dimensional Emotions

| Data Point | Type | Range | Description |
|------------|------|-------|-------------|
| `voice_arousal` | float | -1.0 to 1.0 | Vocal energy/activation |
| `voice_valence` | float | -1.0 to 1.0 | Emotional positivity in voice |

### Pitch Features
Extracted using librosa pitch detection (YIN algorithm).

| Data Point | Type | Unit | Description |
|------------|------|------|-------------|
| `voice_pitch_mean` | float | Hz | Average fundamental frequency |
| `voice_pitch_std` | float | Hz | Pitch variation (standard deviation) |
| `voice_pitch_range` | float | Hz | Difference between max and min pitch |

### Volume/Energy Features
Computed from audio amplitude and RMS energy.

| Data Point | Type | Unit | Description |
|------------|------|------|-------------|
| `voice_volume_mean` | float | dB | Average loudness |
| `voice_volume_std` | float | dB | Volume variation |
| `voice_energy` | float | arbitrary | Overall audio energy |
| `voice_intensity` | float | arbitrary | Sound intensity level |

### Spectral Features
Frequency domain characteristics of the voice.

| Data Point | Type | Unit | Description |
|------------|------|------|-------------|
| `voice_spectral_centroid` | float | Hz | "Brightness" of sound (where frequencies are concentrated) |
| `voice_spectral_rolloff` | float | Hz | Frequency below which 85% of energy exists |
| `voice_zero_crossing_rate` | float | rate | Number of times signal crosses zero (indicates noisiness) |

### MFCC Features (Audio Fingerprint)
Mel-Frequency Cepstral Coefficients - represent the shape of the vocal tract.

| Data Point | Type | Range | Description |
|------------|------|-------|-------------|
| `voice_mfcc_1` through `voice_mfcc_13` | float | varies | 13 cepstral coefficients representing timbre |

**Total MFCC features: 13**

### Prosody Features
Speech pattern characteristics.

| Data Point | Type | Unit | Description |
|------------|------|------|-------------|
| `voice_speaking_rate` | float | syllables/sec | Speed of speech |
| `voice_tempo` | float | BPM | Rhythmic tempo of speech |
| `voice_jitter` | float | % | Pitch instability (emotional stress indicator) |
| `voice_shimmer` | float | % | Amplitude instability (vocal effort indicator) |

---

## 🔗 COMBINED/UNIFIED DATA

When both facial and voice data are available, the system also calculates:

| Data Point | Type | Range | Description |
|------------|------|-------|-------------|
| `combined_arousal` | float | -1.0 to 1.0 | Weighted average of facial and voice arousal |
| `combined_valence` | float | -1.0 to 1.0 | Weighted average of facial and voice valence |

---

## ⏱️ TEMPORAL DATA

| Data Point | Type | Unit | Description |
|------------|------|-------|-------------|
| `timestamp` | float | seconds | Time since start of recording |
| `time_seconds` | float | seconds | Same as timestamp (alias) |
| `frame_number` | int | count | Video frame index (if applicable) |

---

## 📈 SUMMARY STATISTICS

Total unique data points collected:

- **Facial emotions:** 7 basic + 7 dimensional + 2 categorical = **16 features**
- **Voice emotions:** 4 basic + 2 dimensional = **6 features**
- **Voice acoustic:** 3 pitch + 4 volume/energy + 3 spectral = **10 features**
- **Voice MFCC:** **13 features**
- **Voice prosody:** **4 features**
- **Combined:** **2 features**
- **Temporal:** **3 features**

**GRAND TOTAL: 54 unique data points** captured per sample

---

## 🎯 Data Usage in Thesis

### For Predicting Constructive Listening:

**Openness Indicators:**
- High `facial_happy`, `facial_surprise`
- Low `facial_angry`, `facial_fear`
- High `facial_valence` (positive mood)
- Low `voice_pitch_std` (calm, stable voice)
- High `voice_valence`

**Defensiveness Indicators:**
- High `facial_angry`, `facial_disgust`
- Low `facial_arousal` with negative valence ("Depressed" quadrant)
- High `voice_pitch_mean` (stressed voice)
- High `voice_jitter`, `voice_shimmer` (vocal tension)
- Low `voice_valence`

**Genuine Warmth Indicators:**
- High `facial_happy` sustained over time
- Positive `facial_arousal` + positive `facial_valence` ("Excited" quadrant)
- Consistent `voice_happy`
- Moderate `voice_pitch_mean` (natural, not forced)
- Low `voice_jitter` (authentic, not strained)

---

## 📊 Visualization Outputs

### Standard Visualizations
1. **Unified emotions plot** - Facial + voice arousal/valence over time
2. **Facial emotions line plot** - All 7 emotions
3. **Voice features plot** - Pitch, volume, spectral features
4. **Circle movement heatmap** - Arousal-valence space movement
5. **Easy-to-read report** - For non-technical audiences

### Comprehensive Reports (NEW)
6. **Facial comprehensive report** - ALL 16 facial features visualized:
   - All 7 emotions over time
   - Emotion intensity heatmap
   - Arousal & valence timeseries
   - Intensity, excitement, calmness, positivity, negativity
   - Violin plots showing distribution
   - Quadrant distribution
   - Statistical summary table
   - Correlation heatmap

7. **Voice comprehensive report** - ALL 33 voice features visualized:
   - Voice emotions over time
   - Voice arousal & valence
   - Pitch analysis (mean, std, range)
   - Volume analysis (mean, std)
   - Spectral features (centroid, rolloff, zero-crossing)
   - MFCC heatmap (13 coefficients)
   - Prosody features
   - Statistical summary
   - Feature correlations

---

## 💾 Data Storage Format

All data saved to CSV files with structure:

```csv
timestamp,facial_happy,facial_sad,...,voice_happy,voice_pitch_mean,...
0.0,0.234,0.123,...,0.456,187.3,...
0.5,0.267,0.089,...,0.512,192.1,...
...
```

**File naming convention:**
- Camera mode: `webcam_live_emotion_data.csv`
- Video mode: `<filename>_unified_emotion_data.csv`

---

## 🔬 Research Applications

This comprehensive dataset enables:

1. **Multimodal emotion recognition** - Combining facial and vocal cues
2. **Empathy detection** - Tracking emotional congruence between speakers
3. **Stress analysis** - Identifying vocal tension and facial stress
4. **Authenticity detection** - Comparing facial and voice emotional alignment
5. **Temporal emotion dynamics** - Understanding emotion evolution over time
6. **Constructive listening prediction** - ML models using all 54 features

---

*Generated for: Predicting Constructive Listening - A Multimodal Analysis of Empathy*
