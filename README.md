# Empathy Mini-Thesis Pipeline

End‑to‑end multimodal empathy/emotion analysis and prediction pipeline consolidating legacy scripts into a coherent, reproducible structure.

## Project Structure
```
empathy-mini-thesis/
  requirements.txt         - Unified dependency list
  run_pipeline.py          - Master orchestrator (ingestion → grading → analysis → prediction)
  src/
    ingestion/             - Data loading, normalization, unified emotion tracker, fusion
    analysis/              - Group comparison, correlation visuals, summary stats
    grading/               - Transcript content grading & score aggregation
    prediction/            - Feature selection + LOOCV model comparison
  data/
    raw/                   - Unprocessed inputs (transcripts, video/) 
      video/               - Source video files (if available)
      transcripts/         - Source transcript text files
    processed/             - Generated intermediate CSVs (unified, fused, grading outputs)
  outputs/                 - Final plots, heatmaps, model comparison, prediction visuals
  original_archive/        - Clean mirror of historical `0 original` legacy materials
```

## Reorganization & Archived Script
The original heterogeneous code has been preserved verbatim under `original_archive/` for provenance. The one‑time restructuring logic (`reorg_structure.py`) has been archived there; keep it if you may need to regenerate or adjust mappings. For routine usage you no longer need that script at the root.

## Pipeline Overview
1. Ingestion: Synchronizes facial + voice signals; produces unified and fused feature tables.
2. Grading: Evaluates transcript summaries; appends master grading scores.
3. Analysis: Resamples participants, performs group comparisons, correlations, and generates plots.
4. Prediction: Ranks features (Spearman), runs LOOCV across multiple models, selects best performer.

## Quick Start
```powershell
python -m pip install -r requirements.txt
python run_pipeline.py --participant ArianPates --skip-ingestion False --skip-grading False --skip-analysis False --skip-prediction False
```
Generated artifacts will appear in:
- `data/processed/` for intermediate CSVs
- `outputs/` for figures & model result visuals

## Regenerating Structure (If Needed)
If the working folders are accidentally modified or deleted:
1. Retrieve `original_archive/reorg_structure.py`.
2. Run it from project root:
```powershell
python original_archive/reorg_structure.py
```
This recreates data, src, outputs layout and refreshes mirrored legacy files (excluding caches).

## Adding New Participants
Place raw transcripts in `data/raw/transcripts/` and videos (if available) in `data/raw/video/` using consistent participant naming. Re‑run `run_pipeline.py` with the appropriate `--participant` value.

## Notes
- Emotion fusion currently uses facial:voice weight ratio 0.7:0.3.
- Statistical tests include timepoint Kruskal–Wallis; correlations use Spearman.
- Best model selection is based on cross‑validated performance (e.g. MAE, correlation).

## Maintenance
- Keep `requirements.txt` synchronized when adding new media or model libraries.
- Archive rather than delete historical scripts to preserve reproducibility.
- Commit newly generated plots only if they represent stable, published results.