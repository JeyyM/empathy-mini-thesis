# Voice Heatmap Feature Added

## New Menu Structure (Video Mode with Voice)

**10 visualization options:**

1. **Unified analysis** - Facial + voice combined line plots
2. **Facial emotions line plot** - 7 emotions + psychological dimensions
3. **Voice features line plot** - Voice emotions + acoustic features  
4. **Facial emotion heatmap** - Arousal-valence space, intensity, dominant emotions
5. **Voice emotion heatmap** ⭐ NEW - Arousal-valence space, intensity, stress, pitch-volume
6. **Circle movement heatmap** - Facial arousal/valence quadrant movement
7. **Easy-to-read report** - Layperson-friendly summary
8. **All standard visualizations** - Generates options 1-7
9. **Comprehensive reports** - Professional multi-panel reports (facial + voice separate)
10. **EVERYTHING** - All standard + comprehensive = 10 files total

## Voice Heatmap Details

The new voice heatmap (option 5) includes 4 subplots matching the facial heatmap structure:

### Subplot 1: Arousal-Valence Density
- 2D histogram showing distribution of voice emotions in arousal-valence space
- Quadrant labels: Excited (happy), Agitated (angry/fear), Calm (neutral), Depressed (sad)
- Colormap: YlOrRd (yellow-orange-red)

### Subplot 2: Emotional Intensity Distribution  
- Weighted 2D histogram showing average intensity levels
- Same arousal-valence space weighted by voice intensity
- Colormap: Plasma (purple-orange)

### Subplot 3: Stress Level Distribution
- Weighted 2D histogram showing average stress levels
- Arousal-valence space weighted by stress measurements
- Colormap: RdYlGn_r (red-yellow-green reversed - red = high stress)
- **Note:** This replaces the facial heatmap's "Dominant Emotion" subplot

### Subplot 4: Time Evolution ⭐ UPDATED
- 2D histogram showing voice arousal evolution over time
- X-axis: Time (seconds), Y-axis: Arousal level
- Shows how voice arousal changes throughout the recording
- Colormap: Viridis (purple-green-yellow)
- **Matches the facial heatmap's time evolution subplot**

## Implementation

**New method added to `voice_emotion_bot.py`:**
```python
def plot_voice_heatmap(self, save_path=None, bins=20)
```

**Main.py updates:**
- Menu options increased from 9 to 10
- Option 5: Voice heatmap visualization
- All condition checks updated (8, 10 for "all" and "everything")
- Display-only mode also includes voice heatmap option

## File Outputs

### Option 10 (EVERYTHING) now generates:
1. `*_emotion_data.csv` - Raw synchronized data
2. `*_unified_emotions.png` - Unified analysis
3. `*_facial_emotions.png` - Facial line plot
4. `*_voice_features.png` - Voice line plot
5. `*_facial_heatmap.png` - Facial heatmap
6. `*_voice_heatmap.png` - Voice heatmap ⭐ NEW
7. `*_movement_heatmap.png` - Circle movement
8. `*_report.png` - Easy report
9. `*_facial_comprehensive.png` - Facial comprehensive (10 plots)
10. `*_voice_comprehensive.png` - Voice comprehensive (14 plots)

**Total: 10 files (1 CSV + 9 PNGs)**

## Features Visualized

The voice heatmap visualizes these voice features:
- **Emotional dimensions**: arousal, valence, intensity, stress
- **Acoustic features**: pitch_mean, volume_mean (normalized)
- **Spatial distribution**: 2D density in arousal-valence space
- **Relationships**: Pitch-volume correlation with arousal

## Use Case

Perfect for analyzing how voice emotions cluster in emotional space, identifying stress patterns, and understanding the relationship between vocal characteristics (pitch, volume) and emotional arousal.

Example: High stress might cluster in the negative-valence/high-arousal quadrant (Agitated), while calm states cluster in positive-valence/low-arousal (Calm).
