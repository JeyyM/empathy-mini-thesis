# 📊 Complete Visualization Options Guide

## All Available Visualizations in main.py

### 🎤 WITH VOICE DATA (Camera Mode or Video Mode)

When you have both facial and voice data, you get **7 visualization options**:

| Option | Name | Output File(s) | Description |
|--------|------|----------------|-------------|
| **1** | Unified Analysis | `*_unified_emotions.png` | Combined facial + voice arousal/valence timeline |
| **2** | Facial Emotions Only | `*_facial_emotions.png` | Line plot of all 7 facial emotions over time |
| **3** | Voice Features Only | `*_voice_features.png` | Voice pitch, MFCCs, spectral features |
| **4** | Circle Movement Heatmap | `*_movement_heatmap.png` | Arousal-valence circumplex with movement paths |
| **5** | Easy-to-Read Report | `*_report.png` | Layperson-friendly summary with insights |
| **6** | All Standard Visualizations | All of options 1-5 | Generates everything above |
| **7** | 🌟 Comprehensive Reports | `*_facial_comprehensive.png`<br>`*_voice_comprehensive.png` | **NEW!** Professional multi-panel reports with ALL data |

---

### 📸 WITHOUT VOICE DATA (Facial Only Mode)

When you only have facial data, you get **6 visualization options**:

| Option | Name | Output File(s) | Description |
|--------|------|----------------|-------------|
| **1** | Line Plots (Technical) | `*_emotions.png` | All 7 facial emotions timeline |
| **2** | Technical Heatmaps | `*_heatmap.png` | Emotion intensity heatmap |
| **3** | Circle Movement Heatmap | `*_movement_heatmap.png` | Arousal-valence movement visualization |
| **4** | Easy-to-Read Report | `*_report.png` | Layperson-friendly summary |
| **5** | All Standard Visualizations | All of options 1-4 | Generates everything above |
| **6** | 🌟 Comprehensive Report | `*_facial_comprehensive.png` | **NEW!** Professional report with ALL facial data |

---

## 🌟 What's in the Comprehensive Reports?

### Facial Comprehensive Report (16 features, 10 visualizations)

1. ✅ **All 7 Emotions Timeline** - Line plot showing happy, sad, angry, fear, surprise, disgust, neutral
2. ✅ **Emotion Intensity Heatmap** - Color-coded heatmap of all emotions over time
3. ✅ **Arousal & Valence Timeline** - Energy and mood dimensions
4. ✅ **Emotional Intensity** - Overall intensity over time
5. ✅ **Excitement vs Calmness** - Dimensional comparison
6. ✅ **Positivity vs Negativity** - Emotional polarity
7. ✅ **Violin Plots** - Statistical distribution of each emotion
8. ✅ **Quadrant Distribution** - Excited, Agitated, Calm, Depressed states
9. ✅ **Statistical Summary Table** - Mean, std, min, max, median for all emotions
10. ✅ **Correlation Heatmap** - How emotions correlate with each other

### Voice Comprehensive Report (33 features, 14 visualizations)

1. ✅ **Voice Emotions Timeline** - Happy, sad, angry, neutral voice emotions
2. ✅ **Voice Arousal & Valence** - Vocal energy and emotional tone
3. ✅ **Pitch Mean** - Average fundamental frequency over time
4. ✅ **Pitch Variation** - How pitch changes (std deviation)
5. ✅ **Volume Mean** - Loudness levels
6. ✅ **Volume Variation** - Volume stability
7. ✅ **Spectral Centroid** - Voice "brightness"
8. ✅ **Spectral Rolloff** - Frequency distribution
9. ✅ **Zero-Crossing Rate** - Noisiness indicator
10. ✅ **Speaking Rate/Tempo** - Speech speed
11. ✅ **MFCC Heatmap** - 13 coefficients (voice fingerprint)
12. ✅ **Voice Emotion Distribution** - Pie chart of dominant emotions
13. ✅ **Acoustic Features Table** - Statistical summary of all acoustic features
14. ✅ **Prosody Features** - Jitter, shimmer, pitch range (vocal quality)

---

## 📝 Example Usage

### Video Mode with Comprehensive Reports:
```bash
python main.py
> happy.mp4
> Sample rate: 2.0
> Save results? y
> Choose visualization: 7

# Generates:
# - happy_emotion_data.csv
# - happy_facial_comprehensive.png (1.8 MB)
# - happy_voice_comprehensive.png (1.6 MB)
```

### Camera Mode with All Standard Visualizations:
```bash
python main.py
> camera
> Duration: 30
> Sample rate: 1.0
> Save results? y
> Choose visualization: 6

# Generates:
# - webcam_live_emotion_data.csv
# - webcam_live_unified_emotions.png
# - webcam_live_facial_emotions.png
# - webcam_live_voice_features.png
# - webcam_live_movement_heatmap.png
# - webcam_live_report.png
```

---

## 🎯 Which Option to Choose?

| Your Goal | Recommended Option |
|-----------|-------------------|
| Quick overview | Option 1 (Unified) or 5 (Report) |
| Technical analysis | Option 2 (Facial) + 3 (Voice) |
| Publication/thesis | Option 7 (Comprehensive) |
| Present to advisor | Option 5 (Report) |
| Data exploration | Option 6 (All standard) |
| Research appendix | Option 7 (Comprehensive) |
| Conference poster | Option 4 (Heatmap) + 5 (Report) |

---

## 💾 File Sizes (Approximate)

| File Type | Typical Size | Resolution |
|-----------|--------------|------------|
| CSV data | 50-200 KB | N/A |
| Standard visualizations | 100-500 KB | 150 DPI |
| Comprehensive reports | 1.5-4 MB | 300 DPI (publication quality) |

---

## ✅ All Visualizations Are Working!

**Tested and Verified:**
- ✅ Option 1: Unified analysis
- ✅ Option 2: Facial emotions only
- ✅ Option 3: Voice features only
- ✅ Option 4: Circle movement heatmap
- ✅ Option 5: Easy-to-read report
- ✅ Option 6: All standard visualizations
- ✅ Option 7: Comprehensive reports (facial + voice)

**Test Results:**
```
✅ happy_emotion_data.csv (163.8 KB)
✅ happy_facial_comprehensive.png (1819.2 KB)
✅ happy_voice_comprehensive.png (1638.1 KB)
```

---

*Last Updated: November 10, 2025*
*Predicting Constructive Listening: A Multimodal Analysis of Empathy*
