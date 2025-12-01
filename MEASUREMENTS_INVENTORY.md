# Complete List of All Measurements from Emotion Bots

## 📊 Overview
This document lists **every single measurement** that the emotion detection bots (`emotion_bot.py` and `voice_emotion_bot.py`) produce when analyzing a video.

---

## 🎭 FACIAL EMOTION BOT (`emotion_bot.py`)

### **Output: 16 Columns**

#### **Metadata (3 columns)**
1. `timestamp` - Exact datetime when frame was analyzed
2. `time_seconds` - Time in seconds from video start
3. `frame` - Frame number (only in some outputs)

#### **7 Basic Emotions (Probabilities 0-1)**
4. `facial_angry` - Anger/frustration intensity
5. `facial_disgust` - Disgust/revulsion intensity
6. `facial_fear` - Fear/anxiety intensity
7. `facial_happy` - Happiness/joy intensity
8. `facial_sad` - Sadness/sorrow intensity
9. `facial_surprise` - Surprise/shock intensity
10. `facial_neutral` - Neutral/baseline expression

**Note:** These 7 emotions sum to 1.0 (normalized probabilities)

#### **Emotional Dimensions (7 dimensions)**
Based on **Circumplex Model** of affect:

11. `facial_arousal` - Energy/activation level
    - Range: -1 to +1
    - -1 = Very low energy (tired, lethargic)
    - +1 = Very high energy (excited, alert)

12. `facial_valence` - Pleasantness/positivity
    - Range: -1 to +1
    - -1 = Very negative (unpleasant)
    - +1 = Very positive (pleasant)

13. `facial_intensity` - Overall emotional strength
    - Range: 0 to 1
    - 0 = No emotion (neutral)
    - 1 = Very strong emotion

14. `facial_excitement` - Engagement/stimulation level
    - Range: -1 to +1
    - -1 = Very bored/disengaged
    - +1 = Very excited/engaged

15. `facial_calmness` - Inverse of excitement
    - Range: -1 to +1
    - Calculated as: `-excitement`
    - +1 = Very calm/peaceful
    - -1 = Very agitated

16. `facial_positivity` - Overall positive affect
    - Range: -1 to +1
    - +1 = Very positive emotional state
    - -1 = Very negative emotional state

17. `facial_negativity` - Inverse of positivity
    - Range: -1 to +1
    - Calculated as: `-positivity`
    - +1 = Very negative mood
    - -1 = Very positive mood

#### **Emotional State (1 categorical)**
18. `facial_quadrant` - Circumplex quadrant classification
    - **"Excited"** - High arousal (+), Positive valence (+)
        - Energized, happy, enthusiastic
    - **"Stressed"** - High arousal (+), Negative valence (-)
        - Anxious, angry, frustrated
    - **"Calm"** - Low arousal (-), Positive valence (+)
        - Peaceful, content, relaxed
    - **"Tired"** - Low arousal (-), Negative valence (-)
        - Sad, depressed, lethargic

---

## 🎤 VOICE EMOTION BOT (`voice_emotion_bot.py`)

### **Output: 63 Columns**

#### **Metadata (4 columns)**
1. `timestamp` - Exact datetime when segment was analyzed
2. `time_seconds` - Time in seconds from audio start
3. `voice_segment_start` - Sample number where segment starts
4. `voice_segment_end` - Sample number where segment ends

---

### **ACOUSTIC FEATURES (42 columns)**

#### **Pitch Features (4 columns)**
5. `voice_pitch_mean` - Average fundamental frequency (F0) in Hz
    - Typical range: 80-300 Hz
    - Lower = deeper voice, Higher = higher voice

6. `voice_pitch_std` - Standard deviation of pitch
    - Measures pitch variability/fluctuation

7. `voice_pitch_range` - Difference between max and min pitch
    - Wider range = more expressive

8. `voice_pitch_variation` - Coefficient of variation
    - Normalized pitch variability (std/mean)

#### **Volume/Amplitude Features (3 columns)**
9. `voice_volume_mean` - Average loudness/amplitude
    - Range: 0-1 (normalized)
    - Higher = louder voice

10. `voice_volume_std` - Volume variability
    - Measures loudness fluctuation

11. `voice_volume_range` - Volume dynamic range
    - Difference between loudest and quietest

#### **Spectral Features (4 columns)**
12. `voice_spectral_centroid` - Center of mass of spectrum
    - Measures "brightness" of sound
    - Higher = brighter, sharper voice

13. `voice_spectral_centroid_std` - Variability of brightness

14. `voice_spectral_bandwidth` - Spread of frequencies
    - Wider = more complex sound

15. `voice_spectral_rolloff` - Frequency below which 85% of energy lies
    - Indicates voice quality

#### **Voice Quality Features (1 column)**
16. `voice_zero_crossing_rate` - Rate of signal sign changes
    - Indicates noisiness vs harmonicity

#### **MFCCs - Mel-Frequency Cepstral Coefficients (26 columns)**
**THE most important features for speech/emotion recognition**

17-29. `voice_mfcc_1` through `voice_mfcc_13` - Mean MFCCs
    - Capture the spectral envelope (voice timbre)
    - MFCC 1-2: General spectral shape
    - MFCC 3-6: Formant structure (vowel quality)
    - MFCC 7-13: Fine spectral details

30-42. `voice_mfcc_1_std` through `voice_mfcc_13_std` - MFCC variability
    - Measures dynamic changes in voice quality

#### **Prosodic Features (2 columns)**
43. `voice_speech_rate` - Speaking rate estimate
    - Words per minute equivalent
    - Higher = faster speech

44. `voice_silence_ratio` - Proportion of silence
    - Range: 0-1
    - Higher = more pauses

#### **Voice Health/Quality (2 columns)**
45. `voice_harmonic_ratio` - Harmonics vs noise ratio
    - Range: 0-1
    - Higher = clearer, more harmonic voice
    - Lower = breathy, hoarse voice

46. `voice_tremor` - Voice shakiness/tremor
    - Measured in Hz
    - Higher = more trembling voice

---

### **EMOTIONAL DIMENSIONS (5 columns)**

47. `voice_arousal` - Energy/activation in voice
    - Range: -1 to +1
    - -1 = Very low energy (monotone, flat)
    - +1 = Very high energy (animated, loud)
    - **Source:** Wav2vec2 ML model (if available) OR rule-based

48. `voice_valence` - Positive/negative tone
    - Range: -1 to +1
    - -1 = Very negative tone (sad, angry)
    - +1 = Very positive tone (happy, cheerful)
    - **Source:** Wav2vec2 ML model (if available) OR rule-based

49. `voice_dominance` - Assertiveness/confidence
    - Range: -1 to +1
    - -1 = Submissive, weak voice
    - +1 = Dominant, assertive voice
    - **Source:** Wav2vec2 ML model (if available) OR volume-based

50. `voice_intensity` - Overall emotional strength
    - Range: 0 to 1
    - 0 = No emotion (flat affect)
    - 1 = Very strong emotion
    - **Source:** Acoustic features (volume + pitch variation)

51. `voice_stress` - Pressure/tension in voice
    - Range: 0 to 1
    - 0 = Relaxed, calm voice
    - 1 = Very stressed, tense voice
    - **Source:** Pitch variation + harmonic ratio + arousal

---

### **7 Basic Emotions (Probabilities 0-1)**

52. `voice_happy` - Happiness/joy in voice
    - Derived from: High valence + moderate arousal + harmonic voice

53. `voice_angry` - Anger/frustration in voice
    - Derived from: Negative valence + acoustic harshness + pitch variation
    - **Strict detection:** Only if valence < 0.40

54. `voice_sad` - Sadness/sorrow in voice
    - Derived from: Low arousal + negative valence + low volume + silences

55. `voice_fear` - Fear/anxiety in voice
    - Derived from: High arousal + negative valence + tremor + stress

56. `voice_surprise` - Surprise/shock in voice
    - Derived from: Very high arousal + neutral valence + high pitch variation

57. `voice_disgust` - Disgust/revulsion in voice
    - Derived from: Moderate arousal + negative valence + low harmonic ratio

58. `voice_neutral` - Neutral/baseline voice
    - Derived from: Low arousal + neutral valence + stable harmonics

**Note:** These 7 emotions sum to 1.0 (normalized probabilities)

---

## 🔗 COMBINED METRICS (from `unified_emotion_tracker.py`)

When `unified_emotion_tracker.py` synchronizes facial and voice data, it adds:

59. `combined_arousal` - Simple average of facial and voice arousal
    - Formula: `(facial_arousal + voice_arousal) / 2`

60. `combined_valence` - Simple average of facial and voice valence
    - Formula: `(facial_valence + voice_valence) / 2`

61. `combined_intensity` - Simple average of facial and voice intensity
    - Formula: `(facial_intensity + voice_intensity) / 2`

62. `combined_quadrant` - Quadrant based on combined arousal/valence
    - Same categories as facial_quadrant

---

## 📋 COMPLETE COLUMN LIST

### **Total Columns in `[video]_ml_emotion_data.csv`: ~75 columns**

**Temporal (2):**
- `timestamp`
- `time_seconds`

**Facial Emotions (7):**
- `facial_angry`, `facial_disgust`, `facial_fear`, `facial_happy`, `facial_sad`, `facial_surprise`, `facial_neutral`

**Facial Dimensions (7):**
- `facial_arousal`, `facial_valence`, `facial_intensity`, `facial_excitement`, `facial_calmness`, `facial_positivity`, `facial_negativity`

**Facial State (1):**
- `facial_quadrant`

**Voice Segment Info (2):**
- `voice_segment_start`, `voice_segment_end`

**Voice Acoustic - Pitch (4):**
- `voice_pitch_mean`, `voice_pitch_std`, `voice_pitch_range`, `voice_pitch_variation`

**Voice Acoustic - Volume (3):**
- `voice_volume_mean`, `voice_volume_std`, `voice_volume_range`

**Voice Acoustic - Spectral (5):**
- `voice_spectral_centroid`, `voice_spectral_centroid_std`, `voice_spectral_bandwidth`, `voice_spectral_rolloff`, `voice_zero_crossing_rate`

**Voice Acoustic - MFCCs (26):**
- `voice_mfcc_1` through `voice_mfcc_13`
- `voice_mfcc_1_std` through `voice_mfcc_13_std`

**Voice Acoustic - Prosodic (2):**
- `voice_speech_rate`, `voice_silence_ratio`

**Voice Acoustic - Quality (2):**
- `voice_harmonic_ratio`, `voice_tremor`

**Voice Dimensions (5):**
- `voice_arousal`, `voice_valence`, `voice_dominance`, `voice_intensity`, `voice_stress`

**Voice Emotions (7):**
- `voice_happy`, `voice_angry`, `voice_sad`, `voice_fear`, `voice_surprise`, `voice_disgust`, `voice_neutral`

**Combined Metrics (4):**
- `combined_arousal`, `combined_valence`, `combined_intensity`, `combined_quadrant`

---

## 🎯 Summary by Category

| Category | Facial | Voice | Combined | Total |
|----------|--------|-------|----------|-------|
| **Metadata** | 2 | 4 | 0 | 6 |
| **Basic Emotions** | 7 | 7 | 0 | 14 |
| **Dimensions** | 7 | 5 | 3 | 15 |
| **States** | 1 | 0 | 1 | 2 |
| **Acoustic Features** | 0 | 42 | 0 | 42 |
| **TOTAL** | **17** | **58** | **4** | **~75** |

---

## 🔬 Data Types

### **Probabilities (0 to 1)**
- All 7 facial emotions
- All 7 voice emotions
- `facial_intensity`
- `voice_intensity`
- `voice_stress`
- All acoustic features (normalized)

### **Bipolar Scales (-1 to +1)**
- `facial_arousal`, `facial_valence`, `facial_excitement`, `facial_calmness`, `facial_positivity`, `facial_negativity`
- `voice_arousal`, `voice_valence`, `voice_dominance`
- `combined_arousal`, `combined_valence`

### **Categorical**
- `facial_quadrant` (Excited, Stressed, Calm, Tired)
- `combined_quadrant` (Excited, Stressed, Calm, Tired)

### **Continuous (Various Ranges)**
- Pitch features (Hz): 0-500+ Hz
- Volume features (amplitude): 0-1 (normalized)
- Spectral features: Various Hz ranges
- MFCCs: Cepstral coefficients (various ranges)
- Speech rate: Words per minute
- Silence ratio: 0-1
- Harmonic ratio: 0-1
- Tremor: Hz

---

## 💡 Key Insights

### **Most Important Features:**

**For Facial Analysis:**
1. 7 basic emotions (primary output)
2. `facial_arousal` + `facial_valence` (core dimensions)
3. `facial_quadrant` (simplified state)

**For Voice Analysis:**
1. 7 basic emotions (primary output)
2. `voice_arousal` + `voice_valence` (from ML model - 93% accurate)
3. MFCCs 1-13 (most powerful acoustic features)
4. `voice_pitch_mean` + `voice_volume_mean` (intuitive features)

**For Multimodal:**
1. Comparison of facial vs voice emotions
2. `combined_arousal` + `combined_valence` (averaged dimensions)
3. Agreement/disagreement between modalities

---

## 📊 Feature Count Summary

| Type | Count |
|------|-------|
| **Facial Measurements** | 17 |
| **Voice Measurements** | 58 |
| **Combined Measurements** | 4 |
| **Total per Timepoint** | **~75** |

**For 15 participants × 60 seconds × 1Hz sampling:**
- **~67,500 data points** across all measurements
- **~1,350 rows** total (90 per participant)
- **~101,250 total measurements** recorded

---

**End of Measurements Inventory**
