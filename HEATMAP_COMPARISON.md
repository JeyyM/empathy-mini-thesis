# Facial vs Voice Heatmap Comparison

## Structure Comparison

Both heatmaps now have identical 4-subplot layouts for easy comparison:

| Subplot | Facial Heatmap (Option 4) | Voice Heatmap (Option 5) |
|---------|---------------------------|--------------------------|
| **1. Top-Left** | Arousal-Valence Density | Arousal-Valence Density |
| | Shows facial emotion clustering | Shows voice emotion clustering |
| **2. Top-Right** | Intensity Distribution | Intensity Distribution |
| | Weighted by facial intensity | Weighted by voice intensity |
| **3. Bottom-Left** | **Dominant Emotion** | **Stress Level** |
| | Color-coded by dominant facial emotion | Weighted by voice stress level |
| **4. Bottom-Right** | **Time Evolution** | **Time Evolution** |
| | Facial arousal over time | Voice arousal over time |

## Key Similarities ✅

Both heatmaps share:
- Arousal-Valence density visualization
- Intensity distribution mapping
- **Time evolution progression** (showing arousal changes over time)
- Same 2x2 subplot grid layout
- Similar color schemes (coordinated)
- 300 DPI high-quality output

## Key Differences 🔄

### Subplot 3 (Bottom-Left):

**Facial Heatmap:**
- Shows **dominant emotion** by position
- Color-coded scatter plot (7 colors for 7 emotions)
- Answers: "What emotion is most expressed at each arousal-valence coordinate?"

**Voice Heatmap:**
- Shows **stress level** distribution
- Weighted heatmap (red = high stress, green = low stress)
- Answers: "How stressed is the voice at different emotional states?"
- More relevant for voice analysis (stress is measurable in voice features)

## Use Cases

### When to use Facial Heatmap (Option 4):
- Analyzing facial expression patterns
- Understanding which emotions dominate in different arousal-valence regions
- Tracking how facial arousal changes over time
- Visual analysis of facial emotional intensity

### When to use Voice Heatmap (Option 5):
- Analyzing voice emotion patterns
- Understanding stress levels across emotional states
- Tracking how voice arousal changes over time
- Identifying high-stress moments in conversation

### When to use BOTH (Option 10 - EVERYTHING):
- **Multimodal comparison**: See if facial and voice emotions align
- **Stress detection**: Voice stress might reveal hidden tension not visible in face
- **Time synchronization**: Compare facial vs voice arousal evolution
- **Complete picture**: Facial emotions + voice stress + temporal progression

## Example Insights

### Scenario 1: Happy Person
- **Facial heatmap**: Clusters in Excited quadrant (positive valence, high arousal)
- **Voice heatmap**: Also in Excited quadrant, but **low stress** in subplot 3
- **Conclusion**: Genuinely happy (both modalities align, low stress)

### Scenario 2: Anxious Person
- **Facial heatmap**: May show neutral/controlled emotions
- **Voice heatmap**: Clusters in Agitated quadrant with **high stress** in subplot 3
- **Conclusion**: Voice reveals anxiety not visible in face

### Scenario 3: Progressing Emotion
- **Both time evolution plots**: Show arousal rising over time
- **Stress subplot**: Shows increasing stress paralleling arousal
- **Conclusion**: Emotional buildup captured in both modalities

## Technical Details

### Time Evolution (Subplot 4) - Both Heatmaps:
- **X-axis**: Time in seconds
- **Y-axis**: Arousal level (-1 to 1)
- **Color**: Frequency/density (how often that arousal level occurs at that time)
- **Horizontal line**: Arousal = 0 (neutral arousal baseline)
- **Colormap**: Viridis (purple → green → yellow)
- **Purpose**: Show temporal progression of emotional arousal

### Data Source:
- **Facial**: Uses `elapsed_seconds` from facial emotion tracking
- **Voice**: Uses `time_seconds` from voice emotion tracking
- Both synchronized to same timeline when processing video with audio

## Recommended Workflow

1. **Start with Option 1** - Unified analysis to see overall patterns
2. **Use Option 4 & 5** - Compare facial and voice heatmaps side-by-side
3. **Check time evolution** - See if both modalities show similar arousal progression
4. **Analyze stress** - Voice heatmap's stress subplot reveals hidden tension
5. **Use Option 10** - Generate everything for comprehensive analysis and publication

## File Naming

- Facial heatmap: `{prefix}_facial_heatmap.png`
- Voice heatmap: `{prefix}_voice_heatmap.png`

Where `{prefix}` is the video filename (e.g., `happy_facial_heatmap.png`, `happy_voice_heatmap.png`)
