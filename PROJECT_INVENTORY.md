# Empathy Thesis - Complete Project Inventory

**Last Updated:** December 1, 2025  
**Project:** Multimodal Emotion-Cognition Correlation in Empathetic Communication  
**Sample:** n=15-16 participants across 3 groups (Neutral, Opposing, Similar)

---

## 📁 Project Structure

### **Root Directory**
Main working directory containing all analysis scripts and data files.

---

## 🎥 **Video Data Files**

### Test Videos
- `depressed.mp4` - Depression scenario video
- `happy.mp4` - Happy scenario video  
- `testvid.mp4` - Test video for validation

### Emotion Data (CSV)
- `depressed_emotion_data.csv` - Facial emotion analysis
- `depressed.mp4_unified_emotion_data.csv` - Unified multimodal data
- `happy_emotion_data.csv` - Facial emotion analysis
- `happy_voice_emotion_data.csv` - Voice emotion analysis
- `happy.mp4_unified_emotion_data.csv` - Unified multimodal data
- `testvid_emotion_data.csv` - Facial emotion analysis
- `testvid.mp4_unified_emotion_data.csv` - Unified multimodal data
- `webcam_emotion_data.csv` - Live webcam emotion data

---

## 📊 **Core Analysis Data**

### Summary Data (Merged)
Located in `0 original/` folder:
- `facial_summary_merged.csv` - 209 facial features per participant
- `voice_summary_merged.csv` - 375 voice features per participant  
- `fusion_summary_merged.csv` - 273 fusion features per participant

### Grading Data
Located in `text summarization 2/`:
- `grading_results.csv` - Summary quality scores (Overall%, Semantic Similarity, Topic Coverage, Factual Accuracy)

---

## 🤖 **Emotion Detection Systems**

### Main Detection Scripts
1. **`emotion_bot.py`** - Facial emotion detection using FER
   - Detects: angry, disgust, fear, happy, sad, surprise, neutral
   - Outputs: emotion percentages per frame

2. **`voice_emotion_bot.py`** - Voice emotion detection
   - Audio extraction and emotion analysis
   - Detects: emotions from speech patterns

3. **`unified_emotion_tracker.py`** - Multimodal fusion system
   - Combines facial + voice analysis
   - Synchronized emotion tracking

4. **`integrated_emotion_au_bot.py`** - Advanced facial analysis with Action Units
   - AU detection integrated with emotion recognition
   - More granular facial analysis

### Specialized Detectors
- **`au_detector.py`** - Action Unit detection system
- **`webcam_emotion_tracker.py`** - Real-time webcam emotion tracking
- **`quick_webcam_demo.py`** - Quick demo for webcam tracking

### Demo Scripts
- **`run_face_video.py`** - Run facial emotion analysis on videos
- **`main.py`** - Main execution script (original)
- **`main_with_au.py`** - Main execution with AU integration

---

## 📈 **Visualization Scripts**

### Heatmap Generation
1. **`create_heatmap.py`** - Basic emotion heatmap (quadrant-based)
2. **`circle_movement_heatmap.py`** - Circular movement heatmap
3. **`create_voice_heatmap.py`** - Voice emotion heatmap
4. **`create_voice_movement_heatmap.py`** - Combined voice + movement heatmap

### Report Generation
1. **`generate_report.py`** - Facial emotion analysis report
2. **`generate_voice_report.py`** - Voice emotion analysis report

### Generated Visualizations (PNG)
- `depressed_emotions.png`, `depressed_heatmap.png`, `depressed_movement_heatmap.png`, `depressed_report.png`
- `depressed_voice_heatmap.png`, `depressed_voice_movement_heatmap.png`, `depressed_voice_report.png`
- `happy_emotions.png`, `happy_heatmap.png`, `happy_movement_heatmap.png`
- `happy_voice_heatmap.png`, `happy_voice_movement_heatmap.png`, `happy_voice_report.png`
- `testvid_emotions.png`, `testvid_heatmap.png`, `testvid_movement_heatmap.png`, `testvid_report.png`
- `testvid_voice_heatmap.png`, `testvid_voice_movement_heatmap.png`, `testvid_voice_report.png`
- `webcam_emotions.png`, `webcam_emotion_data_heatmap.png`, `webcam_emotion_data_movement_heatmap.png`, `webcam_emotion_data_report.png`

---

## 🔬 **Statistical Analysis**

### Group Comparison Analysis
Located in `0 original/analysis/`:

#### Main Analysis Scripts
1. **`analysis.py`** - Comprehensive statistical analysis framework
   - **Original Functions:**
     - `timepoint_kruskal()` - Test group differences at each time segment
     - `overall_group_tests()` - Session-wide group differences
     - `dominant_emotion_distribution()` - Emotion frequency per group
     - `emotion_change_rate()` & `group_change_rates()` - Volatility analysis
     - `group_emotion_journey()` - Consensus emotional trajectory
   
   - **New Essential Analysis Functions (35 tests):**
     - `analyze_facial_emotions()` - 14 tests (7 emotions + 5 affective + 2 volatility)
     - `analyze_voice_emotions()` - 10 tests (7 emotions + 3 affective)
     - `analyze_fusion_emotions()` - 11 tests (6 affective + 3 agreement + 2 volatility)
     - `analyze_summary_quality()` - 4 tests (Overall%, Semantic, Topic, Factual)
     - `run_essential_analysis()` - Orchestrates all analyses
     - `print_summary_report()` - Formatted output

2. **`run_group_analysis.py`** - Main execution script
   - Runs all 35 statistical tests
   - Generates: `group_comparison_results.csv`

3. **`show_group_comparison.py`** - Detailed results visualization
   - Sorts by p-value
   - Categorizes by significance
   - Shows group means
   - Interprets findings

4. **`create_comparison_table.py`** - Table visualization generator
   - Creates comparison tables with significance indicators
   - Generates: `group_comparison_table.png`, `group_comparison_table_trending.png`

#### Statistical Analysis Outputs
- **`group_comparison_results.csv`** - Complete results (35 tests)
- **`group_comparison_table.png`** - Full comparison table
- **`group_comparison_table_trending.png`** - Trending results (p < 0.15)

#### Statistical Methods
- **Tests:** Kruskal-Wallis (non-parametric ANOVA)
- **Correction:** FDR (Benjamini-Hochberg)
- **Significance:** p < 0.05 (*, **, ***)
- **Trending:** p < 0.10, p < 0.15

#### Key Findings
- **35 tests performed:** 0 significant (p<0.05), 2 trending (p<0.10), 5 marginally trending (p<0.15)
- **Voice Happiness:** p=0.0545 (Opposing > Neutral > Similar)
- **Voice Valence:** p=0.0652 (Similar most negative)
- **Semantic Similarity:** p=0.1054 (Opposing > Similar > Neutral)

---

## 🎯 **Prediction Models**

### Modality-Specific Prediction Scripts
All located in root directory:

1. **`predict_facial_only.py`** - Facial features only
   - 15 features selected via Spearman correlation
   - 6 ML algorithms tested
   - Best: SVR, R²=-0.022, MAE=14.29%
   - Output: `facial_only_predictions.csv`, `facial_only_predictions.png`

2. **`predict_voice_only.py`** - Voice features only
   - 15 features selected via Spearman correlation
   - 6 ML algorithms tested
   - Best: Gradient Boosting, R²=0.492, MAE=10.99%
   - Output: `voice_only_predictions.csv`, `voice_only_predictions.png`

3. **`predict_fusion_only.py`** - Fusion features only
   - 15 features selected via Spearman correlation
   - 6 ML algorithms tested
   - Best: Ridge Regression, R²=0.563, MAE=10.48%
   - Output: `fusion_only_predictions.csv`, `fusion_only_predictions.png`

### Prediction Comparison Scripts
1. **`compare_all_accuracies.py`** - Compare all 4 models
   - Original Fusion vs Facial-Only vs Voice-Only vs Fusion-Only
   - Error distribution: Excellent (≤5%), Good (5-10%), Acceptable (10-15%), Poor (>15%)
   - Output: `all_models_accuracy_comparison.csv`

2. **`show_mean_accuracies.py`** - Group-level accuracy analysis
   - MAE/RMSE by modality and group
   - Prediction bias analysis
   - Best modality per group
   - Output: `modality_group_accuracies.csv`

### Machine Learning Details
- **Validation:** LOOCV (Leave-One-Out Cross-Validation)
- **Algorithms:** Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR
- **Feature Selection:** Spearman correlation (top 15 per modality)
- **Metrics:** MAE, RMSE, r, R², p-values

### Prediction Results Summary
- **Fusion-Only:** Best overall (R²=0.563, MAE=10.48%)
- **Voice-Only:** Highest success rate (53.3% excellent/good predictions)
- **Facial-Only:** Weakest performance (R²=-0.022, MAE=14.29%)
- **Group Patterns:** All models underpredict Opposing, overpredict Similar

---

## 🧪 **Testing & Validation Scripts**

- **`test_quadrants.py`** - Test quadrant-based emotion mapping
- **`test_voice_emotions.py`** - Test voice emotion detection

---

## 📦 **Installation & Setup**

### Installation Scripts
- **`final_install.py`** - Main installation script
- **`install_au_system.py`** - Action Unit system installation

### Requirements Files
- **`requirements.txt`** - Main Python dependencies
- **`voice_requirements.txt`** - Voice analysis dependencies

### Key Dependencies
- `opencv-python` - Video processing
- `fer` - Facial emotion recognition
- `librosa` - Audio analysis
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning
- `matplotlib`, `seaborn` - Visualization
- `scipy`, `statsmodels` - Statistical analysis

---

## 📚 **Documentation**

### Architecture & Guides
- **`ARCHITECTURE.md`** - System architecture documentation
- **`AU_GUIDE.md`** - Action Unit detection guide
- **`AU_INTEGRATION_GUIDE.md`** - AU integration instructions
- **`QUICKSTART.md`** - Quick start guide
- **`SUMMARY.md`** - Project summary

### Analysis Documentation
Located in `0 original/analysis/`:
- **`COMPREHENSIVE_ANALYSIS_OVERVIEW.md`** - Complete analysis inventory
  - Part 1: Original time-series analysis (6 functions)
  - Part 2: Essential group comparison (35 statistical tests)
  - Part 3: Prediction model evaluations (18 models)
  - Summary of all findings and outputs

---

## 📂 **Folder Structure**

```
empathy thesis/
├── 0 original/                          # Original analysis data
│   ├── facial_summary_merged.csv       # 209 facial features
│   ├── voice_summary_merged.csv        # 375 voice features
│   ├── fusion_summary_merged.csv       # 273 fusion features
│   └── analysis/                        # Analysis scripts & results
│       ├── analysis.py                  # Main analysis framework
│       ├── run_group_analysis.py        # Execution script
│       ├── show_group_comparison.py     # Results visualization
│       ├── create_comparison_table.py   # Table generator
│       ├── group_comparison_results.csv # Statistical results
│       ├── group_comparison_table.png   # Full comparison table
│       ├── group_comparison_table_trending.png # Trending results
│       └── COMPREHENSIVE_ANALYSIS_OVERVIEW.md
│   └── analysis_results/                # Original analysis outputs
│       └── groups_summary.json          # Group summary data
├── text summarization 2/                # Summary grading data
│   ├── grading_results.csv             # Quality scores
│   └── exchange.txt                     # Example exchange
├── original facial/                     # Backup of original facial scripts
│   └── [original emotion detection files]
├── __pycache__/                         # Python cache files
├── [Video files: .mp4]                  # Test videos
├── [Emotion data: .csv]                 # Emotion analysis outputs
├── [Visualizations: .png]               # Charts, heatmaps, reports
├── [Detection scripts: .py]             # Emotion detection systems
├── [Analysis scripts: .py]              # Statistical & prediction analysis
├── [Installation scripts: .py]          # Setup files
├── [Requirements: .txt]                 # Dependencies
└── [Documentation: .md]                 # Guides & summaries
```

---

## 🔢 **Data Summary**

### Feature Counts
- **Facial Features:** 209 total
  - 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral)
  - Arousal, valence, dominance
  - Transition rates, volatility metrics
  
- **Voice Features:** 375 total
  - 7 emotions
  - Acoustic features (pitch, volume, speech rate)
  - Arousal, valence, dominance
  
- **Fusion Features:** 273 total
  - Combined facial + voice
  - Modality agreement metrics
  - Cross-modal correlations

### Summary Metrics
- **Overall Summary Score (%)** - Holistic quality
- **Semantic Similarity** - Content relevance
- **Topic Coverage** - Completeness
- **Factual Accuracy** - Correctness

### Participants
- **Total:** n=15 (some analyses n=16 with SamEstose)
- **Neutral Group:** n=5
- **Opposing Group:** n=5
- **Similar Group:** n=5

---

## 🎨 **Visualization Conventions**

### Group Colors
- **Neutral:** #FFB84D (Orange)
- **Opposing:** #4ECDC4 (Teal)
- **Similar:** #FF6B6B (Red)

### Chart Types
- **Bar graphs:** Prediction comparisons with group means
- **Heatmaps:** Emotion intensity over time (quadrant-based)
- **Tables:** Statistical test results with significance indicators
- **Time series:** Emotional trajectories

---

## 🔑 **Key Findings Summary**

### Prediction Models (18 evaluations)
1. **Fusion-Only** best overall: R²=0.563, MAE=10.48%
2. **Voice-Only** highest success rate: 53.3% excellent/good
3. **Facial-Only** weakest: R²=-0.022, MAE=14.29%
4. **Group patterns:** Underpredict Opposing, overpredict Similar

### Group Comparisons (35 statistical tests)
1. **No significant differences** (p<0.05) due to small sample size
2. **2 trending results** (p<0.10):
   - Voice Happiness: Opposing > Neutral > Similar (p=0.0545)
   - Voice Valence: Similar most negative (p=0.0652)
3. **5 marginally trending** (p<0.15):
   - Semantic Similarity, Factual Accuracy, etc.
4. **Pattern:** Opposing group shows higher voice happiness and better summary scores

### Multimodal Integration
- Fusion provides best consistency
- Voice captures group differences better than facial
- Combined modalities improve prediction accuracy

---

## 📊 **Generated Outputs Inventory**

### CSV Files (Data)
- 15+ emotion data files
- 3 summary merged files (facial, voice, fusion)
- 4 prediction output files
- 2 accuracy comparison files
- 1 group comparison results file

### PNG Files (Visualizations)
- 20+ emotion charts
- 15+ heatmaps (various types)
- 10+ report images
- 3+ prediction visualizations
- 2 comparison tables

### Python Scripts
- 10+ detection/tracking scripts
- 8+ visualization scripts
- 5+ analysis scripts
- 3+ prediction scripts
- 3+ installation scripts
- 2+ testing scripts

### Documentation Files
- 7 markdown guides/summaries
- 1 comprehensive analysis overview
- 1 project inventory (this file)

---

## 🚀 **Quick Start Workflow**

### 1. Emotion Detection
```bash
python unified_emotion_tracker.py [video_file]
```

### 2. Generate Visualizations
```bash
python create_heatmap.py
python generate_report.py
```

### 3. Run Statistical Analysis
```bash
cd "0 original/analysis"
python run_group_analysis.py
python show_group_comparison.py
```

### 4. Run Prediction Models
```bash
python predict_facial_only.py
python predict_voice_only.py
python predict_fusion_only.py
python compare_all_accuracies.py
```

### 5. View Results
- Check generated PNG files for visualizations
- Check CSV files for raw data
- Review markdown files for documentation

---

## 📝 **Notes**

- All analysis uses **non-parametric tests** (Kruskal-Wallis) due to small sample size
- **LOOCV** used for prediction to maximize training data
- **FDR correction** applied to control false discovery rate
- Group colors consistent across all visualizations
- File paths handle "0 original" subfolder structure

---

## 🔄 **Version Control**

- **Repository:** empathy-mini-thesis
- **Owner:** JeyyM
- **Branch:** original

---

**End of Inventory** | Total Items Documented: 100+ files, scripts, and outputs
