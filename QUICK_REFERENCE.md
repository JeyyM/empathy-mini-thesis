# 🎯 QUICK REFERENCE: ALL REPORT GENERATORS

## TL;DR - What Should I Use?

```
┌─────────────────────────────────────────────────────────────┐
│  NEED                          →  USE OPTION                │
├─────────────────────────────────────────────────────────────┤
│  Quick overview                →  8 (Easy report)           │
│  Complete analysis             →  11 (EVERYTHING)           │
│  Single mega file              →  12 (ULTIMATE) ⭐          │
│  Research/thesis               →  11 or 12                  │
│  Presentation                  →  12 (ULTIMATE)             │
│  Just facial data              →  4 (Facial heatmap)        │
│  Just voice data               →  5 (Voice heatmap)         │
│  See emotion movement          →  6 or 7 (Movement maps)    │
│  For non-technical audience    →  8 (Easy report)           │
│  Maximum detail separate files →  11 (EVERYTHING)           │
│  Maximum detail one file       →  12 (ULTIMATE) ⭐          │
└─────────────────────────────────────────────────────────────┘
```

## Available Options

### Individual Visualizations (1-8)
```
1. Unified timeline        →  facial + voice combined
2. Facial line plot        →  all facial emotions
3. Voice line plot         →  all voice emotions  
4. Facial heatmap          →  4-panel analysis
5. Voice heatmap           →  4-panel analysis
6. Facial movement         →  circular trajectory
7. Voice movement          →  circular trajectory
8. Easy report             →  layperson-friendly
```

### Aggregate Options (9-12)
```
9.  All standard           →  generates 1-8 (8 files)
10. Comprehensive          →  detailed reports (2 files)
11. EVERYTHING             →  all above (10 files)
12. ULTIMATE MEGA ⭐       →  24+ subplots (1 file)
```

## What Gets Generated

### Option 12 (ULTIMATE) - RECOMMENDED ⭐
```
Output: 1 file (~10 MB)

angry_ULTIMATE_MEGA_REPORT.png
├─ Row 1: All 7 facial emotions timeline
├─ Row 2: All 7 voice emotions timeline  
├─ Row 3: Arousal/valence/intensity/stress comparisons (4 plots)
├─ Row 4: 2D emotion spaces + distributions (4 plots)
├─ Row 5: Correlation matrices (2 plots)
└─ Row 6: Statistics + quadrant analysis (2 plots)

Total: 14 subplots with 100+ data series
```

### Option 11 (EVERYTHING)
```
Output: 10 files (~20 MB total)

1. angry_unified_emotions.png              (~2 MB)
2. angry_facial_emotions.png               (~2 MB)
3. angry_voice_features.png                (~2 MB)
4. angry_facial_heatmap.png                (~2 MB)
5. angry_voice_heatmap.png                 (~2 MB)
6. angry_movement_heatmap.png              (~2 MB)
7. angry_voice_movement_heatmap.png        (~2 MB)
8. angry_report.png                        (~3 MB)
9. angry_facial_comprehensive.png          (~3 MB)
10. angry_voice_comprehensive.png          (~3 MB)
```

## Usage Examples

### For Thesis (RECOMMENDED)
```bash
python main.py
> angry.mp4
> 1.0
> y
> 12  ← ULTIMATE MEGA REPORT
```
**Result**: One comprehensive figure with all data

### For Complete Archive
```bash
python main.py  
> angry.mp4
> 0.5  ← More frequent sampling
> y
> 11  ← EVERYTHING
```
**Result**: 10 separate visualization files

### For Quick Check
```bash
python main.py
> angry.mp4
> 1.0
> y
> 8  ← Easy report
```
**Result**: Simple layperson-friendly summary

## File Sizes

```
Option  Files  Total Size   Per File
─────────────────────────────────────
  1      1      ~2 MB       2 MB
  2      1      ~2 MB       2 MB
  3      1      ~2 MB       2 MB
  4      1      ~2 MB       2 MB
  5      1      ~2 MB       2 MB
  6      1      ~2 MB       2 MB
  7      1      ~2 MB       2 MB
  8      1      ~3 MB       3 MB
  9      8     ~15 MB       2 MB avg
  10     2      ~6 MB       3 MB avg
  11    10     ~20 MB       2 MB avg
  12     1     ~10 MB      10 MB ⭐
```

## Processing Time

```
Option  Time    Files Generated
────────────────────────────────
 1-8    2-3s    1 file
  9     10s     8 files
  10    5s      2 files
  11    15s     10 files
  12    8s      1 file ⭐
```

## Best Combinations

### Thesis Defense
```
Option 12 + CSV export
→ One mega figure + raw data for questions
```

### Publication
```  
Option 12 (main figure) + Option 11 (supplementary)
→ Comprehensive main + detailed supplementary
```

### Quick Analysis
```
Option 8 or 12
→ Either simple or comprehensive, both in one file
```

### Archive Everything
```
Option 11 + CSV
→ All visualizations + raw data for future use
```

## Decision Tree

```
Start
  │
  ├─ Need ONE file? ────────────────→ Option 12 ⭐
  │
  ├─ Need SEPARATE files? ──────────→ Option 11
  │
  ├─ Quick overview only? ──────────→ Option 8
  │
  ├─ Just facial analysis? ─────────→ Option 4
  │
  ├─ Just voice analysis? ──────────→ Option 5
  │
  └─ Want everything possible? ─────→ Option 11 or 12 ⭐
```

## Always Generated

Regardless of option chosen:
```
✅ CSV file with ALL raw data
   - angry_emotion_data.csv
   - Contains every emotion, dimension, feature
   - Timestamp for each sample
   - Can load into Excel, Python, R, etc.
```

## Recommended Workflow

### Step 1: Process Video
```bash
python main.py
> your_video.mp4
> 1.0  # or 0.5 for more detail
> y
```

### Step 2: Choose Output
```bash
> 12  ← For single comprehensive file
or
> 11  ← For all separate files
```

### Step 3: Review Results
- Open the PNG file(s)
- Check CSV in Excel if needed
- Use for analysis/presentation

## Summary

```
┌────────────────────────────────────────────────────┐
│  🏆 TOP RECOMMENDATIONS                            │
├────────────────────────────────────────────────────┤
│  1. Option 12 - ULTIMATE MEGA REPORT ⭐            │
│     → Single comprehensive 24-subplot figure       │
│     → Best for: Thesis, publications, presentations│
│                                                     │
│  2. Option 11 - EVERYTHING                         │
│     → All 10 visualization files separate          │
│     → Best for: Archive, detailed analysis         │
│                                                     │
│  3. Option 8 - Easy Report                         │
│     → Simple layperson-friendly summary            │
│     → Best for: Quick checks, non-technical viewers│
└────────────────────────────────────────────────────┘
```

**Bottom Line**: Use Option 12 (ULTIMATE MEGA REPORT) for maximum insight in a single file! 🚀
