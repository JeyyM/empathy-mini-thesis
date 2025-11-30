Analysis package for multimodal emotion outputs

Usage

1. Install dependencies (preferably in a virtualenv):

```powershell
python -m pip install -r analysis/requirements.txt
```

2. Run the analysis:

```powershell
python analysis/run_analysis.py --results-dir results --out-dir analysis_results --n-segments 100
```

Outputs will be saved in `analysis_results` including JSON summaries and PNG plots.

Design notes
- Each participant is resampled to `n_segments` so every participant contributes equally.
- Timepoint-wise Kruskal-Wallis tests are performed across groups and FDR-corrected when possible.
