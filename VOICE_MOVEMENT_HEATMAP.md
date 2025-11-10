# Voice Movement Heatmap Feature Added

## New Complete Menu Structure (Video Mode with Voice)

**11 visualization options:**

1. **Unified analysis** - Facial + voice combined line plots
2. **Facial emotions line plot** - 7 emotions + psychological dimensions
3. **Voice features line plot** - Voice emotions + acoustic features  
4. **Facial emotion heatmap** - Arousal-valence space, intensity, dominant emotions, time evolution
5. **Voice emotion heatmap** - Arousal-valence space, intensity, stress, time evolution
6. **Facial movement heatmap** - Circle showing facial emotion movement patterns
7. **Voice movement heatmap** ⭐ NEW - Circle showing voice emotion movement patterns
8. **Easy-to-read report** - Layperson-friendly summary
9. **All standard visualizations** - Generates options 1-8
10. **Comprehensive reports** - Professional multi-panel reports (facial + voice separate)
11. **EVERYTHING** - All standard + comprehensive = 11 files total

## Voice Movement Heatmap Details

The new voice movement heatmap (option 7) creates a **2-panel circular visualization**:

### Panel 1: Where Voice Spends Time (Movement Density)
- 2D histogram within emotion circle showing time spent in each quadrant
- **Quadrants:**
  - **Top-Right (Excited/Energized)**: Positive mood + High energy
  - **Top-Left (Stressed/Anxious)**: Negative mood + High energy
  - **Bottom-Right (Peaceful/Relaxed)**: Positive mood + Low energy
  - **Bottom-Left (Tired/Low mood)**: Negative mood + Low energy
- Color intensity shows frequency (how often voice is in that region)
- Colormap: YlOrRd (yellow-orange-red)

### Panel 2: Voice Emotional Journey (Movement Path)
- Gaussian-smoothed trajectory showing voice emotion progression
- **Green circle**: Starting point
- **Red circle**: Ending point
- White trajectory line showing the path traveled
- Intensity shows concentration of time along the path
- Colormap: Plasma (purple-orange-yellow)

## Comparison: Facial vs Voice Movement Heatmaps

| Feature | Facial Movement (Option 6) | Voice Movement (Option 7) |
|---------|----------------------------|---------------------------|
| **Data Source** | Facial arousal/valence | Voice arousal/valence |
| **Panel 1** | Where face spends time | Where voice spends time |
| **Panel 2** | Facial emotional journey | Voice emotional journey |
| **Use Case** | Visual expression patterns | Vocal emotion patterns |
| **Insights** | Where facial emotions cluster | Where voice emotions cluster |

## Complete File Outputs

### Option 11 (EVERYTHING) now generates 11 files:
1. `*_emotion_data.csv` - Raw synchronized data
2. `*_unified_emotions.png` - Unified analysis
3. `*_facial_emotions.png` - Facial line plot
4. `*_voice_features.png` - Voice line plot
5. `*_facial_heatmap.png` - Facial heatmap (4 subplots)
6. `*_voice_heatmap.png` - Voice heatmap (4 subplots)
7. `*_movement_heatmap.png` - Facial movement (2 panels)
8. `*_voice_movement_heatmap.png` - Voice movement ⭐ NEW (2 panels)
9. `*_report.png` - Easy report
10. `*_facial_comprehensive.png` - Facial comprehensive (10 plots)
11. `*_voice_comprehensive.png` - Voice comprehensive (14 plots)

**Total: 11 files (1 CSV + 10 PNGs)**

## Implementation Details

**New method in `voice_emotion_bot.py`:**
```python
def plot_voice_movement_heatmap(self, save_path=None, grid_size=50)
```

**Features:**
- Circular boundary masking (only shows data within emotion circle)
- Gaussian smoothing for trajectory visualization
- Start/end markers (green/red)
- Quadrant labels with color-coded boxes
- Equal aspect ratio for accurate circle representation
- 300 DPI output for publication quality

## Use Cases

### Individual Analysis:
- **Option 7**: See voice movement patterns in isolation
- **Option 6**: See facial movement patterns in isolation

### Comparative Analysis:
- **Option 11**: Generate both movement heatmaps to compare
- Check if facial and voice emotions follow similar paths
- Identify discrepancies (e.g., calm face but stressed voice)

### Research Applications:
- **Empathy analysis**: Do empathetic responses cluster in specific quadrants?
- **Stress detection**: Does voice spend more time in stressed quadrant than face?
- **Emotional congruence**: Do facial and voice emotions align spatially?
- **Temporal patterns**: Do both modalities progress similarly over time?

## Example Interpretation

### Happy Video:
- **Facial movement**: Clusters in Excited quadrant (top-right)
- **Voice movement**: Also clusters in Excited quadrant
- **Conclusion**: Genuine happiness (multimodal agreement)

### Depressed Video:
- **Facial movement**: Clusters in Tired quadrant (bottom-left)
- **Voice movement**: Also in Tired quadrant with some movement to Stressed
- **Conclusion**: Low mood with occasional stress spikes

### Anxious Person:
- **Facial movement**: May show Calm/Peaceful (controlled expression)
- **Voice movement**: Clusters in Stressed quadrant (high arousal, negative valence)
- **Conclusion**: Voice reveals anxiety hidden in facial expressions

## Testing

To test the new voice movement heatmap:

```bash
python main.py
happy.mp4
10
y
7
```

This will generate `happy_voice_movement_heatmap.png` showing:
- Left panel: Density heatmap of where voice emotions cluster
- Right panel: Trajectory showing emotional journey from start (green) to end (red)

Or use **Option 11** to generate EVERYTHING including both movement heatmaps for comparison!

## Technical Notes

- Grid size: 50x50 bins for smooth visualization
- Gaussian sigma: 3 (controls trajectory smoothing)
- Circular mask: Only displays data where x² + y² ≤ 1
- Axis limits: -1.1 to 1.1 (allows space for labels outside circle)
- Figure size: 16x8 inches (2 panels side-by-side)
- DPI: 300 (high resolution for publication)
