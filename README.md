# Multimodal Empathy Mini-Thesis

End-to-end pipeline for extracting facial and voice emotions from video, fusing modalities, grading human summaries against transcripts, running correlation analyses, and predicting listening/summary performance.

## Overview
- Extracts facial and voice emotion timelines from video, aligned on a unified timeline.
- Performs multimodal fusion (70% facial, 30% voice) with derived dimensions and states.
- Grades summaries vs transcripts and stores labels for analysis and prediction.
- Runs correlation analyses and scatter visualizations across modalities.
- Predicts Listening/Summary score from fused features and merges Actual labels when available.

## Quickstart
```powershell
# 1) Install dependencies
pip install -r .\requirements.txt

# 2) Run per-participant ML analysis (interactive prompts)
python .\src\analysis\run_multimodal_emotion.py

# 3) Batch grade all transcript/summary pairs
python .\src\summary_assessment\run_all_grading.py

# 4) Correlation analysis (fusion features vs summary grading)
python .\src\correlation\fusion_summary_correlation.py

# 5) Scatter visualizations (facial, voice, fusion)
python .\src\correlation\visualize_correlations_scatter.py

# 6) Run fusion-only prediction model
python .\src\prediction\predict_fusion_only.py
```

## Repository Structure (key folders)
- `src/analysis/`
	- `unified_emotion_tracker.py`: unified extraction (facial + voice ML) → CSV
	- `fusion.py`: multimodal fusion with derived metrics and states
	- `facial_reports.py`, `voice_reports.py`, `fusion_reports.py`: visualizations
	- `run_multimodal_emotion.py`: interactive per-participant analysis → `results/<Name>/`
- `src/summary_assessment/`
	- `run_assessment.py`: grade a single transcript/summary pair
	- `run_all_grading.py`: batch grade all pairs → `data/grading_results.csv`
- `src/correlation/`
	- `fusion_summary_correlation.py`: merges fusion stats with grades; correlations
	- `visualize_correlations_scatter.py`: scatter plots and summary CSV
- `src/prediction/`
	- analysis utilities and modality comparisons; pipeline invokes fusion-only model
- `data/`
	- `transcript/<neutral|opposing|similar>/NameChat.txt` and `NameSummary.txt`
	- `processed/master_summary_scores.csv` (pipeline Step B labels)
	- `grading_results.csv` (batch grading output)
- `results/<Name>/`: per-participant CSVs and PNGs from analysis
- `output`
	- `correlation_analysis/`: all correlation figures and CSVs
	- `prediction_analysis/`: prediction outputs (`fusion_only_predictions.csv`)

## Data Layout
- Videos: place under `data/raw/video/` (or provide a full path to the script).
- Transcripts: `data/transcript/<Group>/NameChat.txt`
- Summaries: `data/transcript/<Group>/NameSummary.txt`
- Labels:
	- Batch grading writes `data/grading_results.csv`.
	- Pipeline Step B appends `data/processed/master_summary_scores.csv`.

## Per-Participant Analysis (writes to results/<Name>/)
Interactive pipeline to extract, fuse, and visualize ML-enhanced features.
```powershell
python .\src\analysis\run_multimodal_emotion.py
```
Outputs in `results/<Name>/`:
- `<Name>_ml_emotion_data.csv` (unified facial + ML voice timeline)
- `<Name>_ml_fusion.csv` (fused + derived metrics and states)
- PNGs: facial/voice/fusion emotions, dimensions, states, acoustic

## Grading (Summary Assessment)
- Batch grading across all groups:
```powershell
python .\src\summary_assessment\run_all_grading.py
```
- Single pair grading:
```powershell
python .\src\summary_assessment\run_assessment.py data\transcript\neutral\MiguelBorromeoChat.txt data\transcript\neutral\MiguelBorromeoSummary.txt --name MiguelBorromeo --group Neutral
```
Outputs:
- `data/grading_results.csv` (batch)
- `data/processed/master_summary_scores.csv` (pipeline Step B)

## Correlation Analysis (centralized outputs)
```powershell
# Fusion vs summary: merged table, correlations, heatmap
python .\src\correlation\fusion_summary_correlation.py

# Scatter plots across modalities + summary CSV
python .\src\correlation\visualize_correlations_scatter.py
```
Writes to `outputcorrelation_analysis/`:
- `fusion_summary_merged.csv`
- `fusion_summary_correlations.csv`
- `fusion_summary_correlation_heatmap.png`
- `scatter_core_dimensions.png`, `scatter_emotions_page{1..3}.png`
- `scatter_volatility.png`, `scatter_transitions.png`, `scatter_voice_acoustic.png`
- `scatter_correlations_summary.csv`

## Prediction (fusion-only)
- Run the fusion-only model:
```powershell
python .\src\prediction\predict_fusion_only.py
```
- Outputs in current working directory: `fusion_only_predictions.csv`, `fusion_only_predictions.png`
- Note: `src\prediction\analyze_accuracy.py` expects `output\prediction_analysis\individual_predictions.csv` (legacy path used by earlier runs)

## Troubleshooting
- Paths on Windows: prefer quoted paths in PowerShell; the code now centralizes outputs to avoid scattered files.
- Missing CSVs for correlation: scripts auto-resolve common paths; if missing, re-run grading and correlation steps.
- Summary import errors: run from repo root. If needed, use `python -m summary_assessment.run_all_grading` style imports.
- Audio extraction verbosity: the unified tracker uses MoviePy with `logger=None` to avoid unsupported verbose arg issues.

## Notes
- Tested on Windows PowerShell. Figures are saved as PNG; CSVs use UTF-8.
- Exact library versions are pinned in `requirements.txt`.

## Contributors
- Ewan Rafael A. Escano
- Gerard Christian A. Felipe
- Andre Gabriel D. Llanes
- Juan Miguel B. Miranda
