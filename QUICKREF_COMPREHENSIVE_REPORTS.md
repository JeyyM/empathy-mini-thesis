# 📋 QUICK REFERENCE - Comprehensive Reports

## 🚀 Quick Start

### Generate Reports from main.py
```bash
python main.py
> camera (or video file)
> Duration: 30
> Save? y
> Visualization: 7  # ← Choose this for comprehensive reports!
```

### Generate Reports from CSV
```bash
python generate_all_reports.py your_data.csv
```

---

## 📊 Report Outputs

| Report Type | Features | Visualizations | File Size |
|-------------|----------|----------------|-----------|
| **Facial** | 16 | 10 plots | ~2-3 MB |
| **Voice** | 33 | 14 plots | ~3-4 MB |

---

## 🎯 What Gets Visualized

### Facial Report (16 features)
- 7 basic emotions
- Arousal, valence, intensity
- Excitement, calmness
- Positivity, negativity
- Quadrant analysis
- Statistics & correlations

### Voice Report (33 features)
- 4 voice emotions
- Voice arousal & valence
- Pitch (mean, std, range)
- Volume (mean, std, energy, intensity)
- Spectral (centroid, rolloff, ZCR)
- 13 MFCC coefficients
- Prosody (jitter, shimmer, tempo, rate)

---

## 📁 File Structure

```
your_data.csv                           # Raw data
your_data_facial_comprehensive.png     # Facial report
your_data_voice_comprehensive.png      # Voice report
```

---

## 🎨 Customization

Edit report scripts to customize:
- Colors: Change `color='#FF6B6B'`
- Resolution: Change `dpi=300`
- Size: Change `figsize=(24, 16)`
- Layout: Modify `GridSpec(rows, cols)`

---

## 💡 Pro Tips

✅ **Record 30+ seconds** for meaningful statistics
✅ **Good lighting** for facial detection
✅ **Clear audio** for voice features
✅ **Active expressions** for interesting visualizations
✅ **Check CSV first** to verify data quality

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No facial data | Check for `facial_*` columns in CSV |
| No voice data | Only available in camera/unified mode |
| Reports empty | Need 10+ samples minimum |
| File too large | Reduce DPI in scripts |

---

## 📚 Documentation

- **Full guide:** `COMPREHENSIVE_REPORTS_GUIDE.md`
- **Data reference:** `DATA_COLLECTION_REFERENCE.md`
- **Implementation:** `COMPREHENSIVE_REPORTS_COMPLETE.md`

---

## ✅ Checklist

Before generating reports:
- [ ] CSV file exists and has data
- [ ] At least 10 samples recorded
- [ ] Columns include facial_* or voice_* features
- [ ] Duration is 5+ seconds

After generating:
- [ ] PNG files created successfully
- [ ] Open files to verify appearance
- [ ] Check resolution (should be sharp)
- [ ] Verify all plots populated

---

*Quick Reference v1.0 - Comprehensive Reports System*
