#!/usr/bin/env python3
"""
Master pipeline to run the sequential Multimodal Empathy Prediction workflow.

Sequence:
  A) Ingestion & Extraction: Process participant video -> fused features
  B) Grading: Grade transcript vs summary -> append to master_summary_scores.csv
  C) Data Fusion & Correlation: Run group comparisons and correlation visuals
  D) Prediction: Run fusion-only model and report participant's predicted vs actual

Usage examples (PowerShell):
  python run_pipeline.py --name MiguelNg --video "data/raw/MiguelNg.mp4" \
    --transcript "data/raw/transcripts/opposing/MiguelNgChat.txt" \
    --summary "data/raw/transcripts/opposing/MiguelNgSummary.txt" --group Opposing

You can also run interactively without arguments; the script will prompt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"

# Original folders (for backward-compatible calls)
ORIG_DIR = BASE_DIR / "0 original"
ORIG_ANALYSIS = ORIG_DIR / "analysis"
ORIG_CORRELATION = ORIG_DIR / "correlation"
ORIG_PREDICTION = ORIG_DIR / "prediction"
ORIG_GRADING = ORIG_DIR / "text summarization 2"

# Ensure essential directories exist
for d in [DATA_RAW, DATA_PROCESSED, OUTPUTS]:
    d.mkdir(parents=True, exist_ok=True)


def _add_sys_paths():
    # Allow importing from src/ tree and original code when needed
    for p in [
        SRC_DIR,
        SRC_DIR / "ingestion",
        SRC_DIR / "grading",
        SRC_DIR / "analysis",
        SRC_DIR / "prediction",
        ORIG_DIR,
        ORIG_ANALYSIS,
        ORIG_CORRELATION,
        ORIG_PREDICTION,
        ORIG_GRADING,
    ]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def find_video_for_name(name: str) -> Optional[Path]:
    if not DATA_RAW.exists():
        return None
    patterns = [
        f"{name}.mp4",
        f"{name}.mov",
        f"{name}.mkv",
        f"{name}*.mp4",
        f"{name}*.mov",
        f"{name}*.mkv",
    ]
    # search recursively
    for root, _, files in os.walk(DATA_RAW):
        for f in files:
            for pat in patterns:
                if Path(f).match(pat):
                    return Path(root) / f
    return None


def step_a_ingestion(
    name: str, video_path: Optional[Path], sample_interval: float = 1.0
) -> Optional[Path]:
    print("\n=== Step A: Ingestion & Extraction ===")
    _add_sys_paths()

    if not video_path:
        video_path = find_video_for_name(name)
    if not video_path or not Path(video_path).exists():
        print(f"[warn] Video not found for participant '{name}'. Skipping Step A.")
        return None

    print(f"[info] Using video: {video_path}")
    # Import UnifiedEmotionTracker strictly from src ingestion (no fallback)
    from ingestion.unified_emotion_tracker import UnifiedEmotionTracker  # type: ignore

    from ingestion.fusion import MultimodalEmotionFusion  # type: ignore

    # Process video -> unified df
    tracker = UnifiedEmotionTracker(sample_rate=22050, use_ml_voice=True)
    df_unified = tracker.process_video_unified(
        str(video_path), sample_interval=sample_interval
    )
    if df_unified is None or df_unified.empty:
        print("[warn] Unified extraction returned no data. Skipping Step A.")
        return None

    unified_csv = DATA_PROCESSED / f"{name}_unified.csv"
    df_unified.to_csv(unified_csv, index=False)
    print(f"[ok] Unified features -> {unified_csv}")

    # Fuse -> features
    fusion = MultimodalEmotionFusion(w_facial=0.7, w_voice=0.3)
    fused_df = fusion.fuse_data(
        str(unified_csv), output_csv=str(DATA_PROCESSED / f"{name}_features.csv")
    )

    # Collect any generated plot files into outputs (best-effort)
    for fn in Path(".").glob(f"{name}_ml_*.*"):
        try:
            shutil.copy2(str(fn), str(OUTPUTS / fn.name))
        except Exception:
            pass

    features_csv = DATA_PROCESSED / f"{name}_features.csv"
    print(f"[ok] Fused features -> {features_csv}")
    return features_csv


def step_b_grading(
    name: str,
    transcript_file: Optional[Path],
    summary_file: Optional[Path],
    group: Optional[str],
) -> Optional[pd.Series]:
    print("\n=== Step B: Grading (Summary Score) ===")
    _add_sys_paths()

    # Resolve transcript/summary if not provided
    if not transcript_file or not Path(transcript_file).exists():
        # Try common layout under data/raw/transcripts
        base = DATA_RAW / "transcripts"
        candidates = []
        if base.exists():
            for root, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith("chattxt") or f.endswith("Chat.txt"):
                        if f.startswith(name):
                            candidates.append(Path(root) / f)
        transcript_file = candidates[0] if candidates else None

    if not summary_file or not Path(summary_file).exists():
        base = DATA_RAW / "transcripts"
        candidates = []
        if base.exists():
            for root, _, files in os.walk(base):
                for f in files:
                    if f.lower().endswith("summary.txt"):
                        if f.startswith(name):
                            candidates.append(Path(root) / f)
        summary_file = candidates[0] if candidates else None

    if (
        not transcript_file
        or not summary_file
        or not Path(transcript_file).exists()
        or not Path(summary_file).exists()
    ):
        print(f"[warn] Missing transcript/summary for '{name}'. Skipping Step B.")
        return None

    # Import grader
    try:
        from grading.content_grader import ContentRecallGrader  # type: ignore
    except Exception:
        sys.path.insert(0, str(ORIG_GRADING))
        from main import ContentRecallGrader  # type: ignore

    grader = ContentRecallGrader()
    result = grader.grade_summary(str(transcript_file), str(summary_file))
    if "error" in result:
        print(f"[warn] Grading failed: {result['error']}")
        return None

    row = {
        "Name": name,
        "Group": (group or "").title() if group else "",
        "Overall_Percentage": result.get("overall_percentage"),
        "Letter_Grade": result.get("letter_grade"),
        "Semantic_Similarity": result["breakdown"]["semantic_similarity"]["score"],
        "Topic_Coverage": result["breakdown"]["topic_coverage"]["score"],
        "Factual_Accuracy": result["breakdown"]["factual_accuracy"]["score"],
        "Original_Words": result["summary_stats"]["original_words"],
        "Summary_Words": result["summary_stats"]["summary_words"],
        "Compression_Ratio": result["summary_stats"]["compression_ratio"],
    }

    out_csv = DATA_PROCESSED / "master_summary_scores.csv"
    df = pd.DataFrame([row])
    if out_csv.exists():
        df.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)
    print(f"[ok] Appended grading -> {out_csv}")
    return pd.Series(row)


def step_c_correlation():
    print("\n=== Step C: Data Fusion & Correlation ===")
    _add_sys_paths()
    # 1) Run essential group analysis to generate group_comparison_results.csv
    try:
        # Prefer the copied module under src/, fall back to original by path import
        try:
            from analysis.group_analysis import run_essential_analysis  # type: ignore
        except Exception:
            try:
                # Explicitly try src-prefixed import if available
                from src.analysis.group_analysis import run_essential_analysis  # type: ignore
            except Exception:
                # Last resort: import directly from file path (src), then (original)
                import importlib.util

                src_ga = SRC_DIR / "analysis" / "group_analysis.py"
                spec = None
                if src_ga.exists():
                    spec = importlib.util.spec_from_file_location(
                        "group_analysis", str(src_ga)
                    )
                else:
                    orig_ga = ORIG_ANALYSIS / "analysis.py"
                    if orig_ga.exists():
                        spec = importlib.util.spec_from_file_location(
                            "group_analysis", str(orig_ga)
                        )
                if spec is None or spec.loader is None:
                    raise ImportError(
                        "Cannot resolve group_analysis module from src or original"
                    )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                run_essential_analysis = getattr(mod, "run_essential_analysis")

        print("[info] Running essential group analysis...")
        # Use original root where merged CSVs and grading CSV live
        run_essential_analysis(str(ORIG_DIR))
    except Exception as e:
        print(f"[warn] run_essential_analysis failed: {e}")

    # 2) Create comparison tables (cwd=ORIG_ANALYSIS)
    try:
        print("[info] Generating comparison tables...")
        os.chdir(str(ORIG_ANALYSIS))
        # Execute the script module-style
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "create_comparison_table", str(ORIG_ANALYSIS / "create_comparison_table.py")
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        # collect PNGs to outputs
        for f in ["group_comparison_table.png", "group_comparison_table_trending.png"]:
            p = ORIG_ANALYSIS / f
            if p.exists():
                shutil.copy2(str(p), str(OUTPUTS / p.name))
    except Exception as e:
        print(f"[warn] comparison table generation failed: {e}")
    finally:
        os.chdir(str(BASE_DIR))

    # 3) Correlation scatter visuals (cwd=ORIG_CORRELATION)
    try:
        print("[info] Generating correlation scatter visuals...")
        os.chdir(str(ORIG_CORRELATION))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "visualize_correlations_scatter",
            str(ORIG_CORRELATION / "visualize_correlations_scatter.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        # copy generated outputs
        for f in [
            "scatter_core_dimensions.png",
            "scatter_emotions_page1.png",
            "scatter_emotions_page2.png",
            "scatter_emotions_page3.png",
            "scatter_volatility.png",
            "scatter_transitions.png",
            "scatter_voice_acoustic.png",
            "scatter_correlations_summary.csv",
        ]:
            p = ORIG_CORRELATION / f
            if p.exists():
                shutil.copy2(str(p), str(OUTPUTS / p.name))
    except Exception as e:
        print(f"[warn] correlation visuals failed: {e}")
    finally:
        os.chdir(str(BASE_DIR))


def step_d_prediction(name: str):
    print("\n=== Step D: Prediction (Fusion-Only Model) ===")
    _add_sys_paths()
    # Run the fusion-only model in its expected working directory
    try:
        os.chdir(str(ORIG_PREDICTION))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "predict_fusion_only", str(ORIG_PREDICTION / "predict_fusion_only.py")
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[warn] prediction run failed: {e}")
    finally:
        os.chdir(str(BASE_DIR))

    # Copy outputs and print participant row if present
    pred_csv = ORIG_PREDICTION / "fusion_only_predictions.csv"
    if pred_csv.exists():
        shutil.copy2(str(pred_csv), str(OUTPUTS / pred_csv.name))
        try:
            df = pd.read_csv(pred_csv)
            row = df[df["Name"].astype(str).str.lower() == name.lower()]
            if not row.empty:
                r = row.iloc[0]
                print(
                    f"Predicted Listening Score: {r['Predicted']:.1f}% | Actual Score: {r['Actual']:.1f}%"
                )
            else:
                print(f"[info] Participant '{name}' not found in prediction results.")
        except Exception:
            pass
    else:
        print("[info] No prediction CSV produced.")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run the Multimodal Empathy Prediction pipeline"
    )
    ap.add_argument("--name", help="Participant name (e.g., MiguelNg)")
    ap.add_argument("--video", help="Path to participant video file", default=None)
    ap.add_argument(
        "--transcript", help="Path to participant transcript (Chat.txt)", default=None
    )
    ap.add_argument(
        "--summary", help="Path to participant summary (Summary.txt)", default=None
    )
    ap.add_argument(
        "--group",
        help="Experimental group for participant (Neutral/Opposing/Similar)",
        default=None,
    )
    ap.add_argument("--skip-ingestion", action="store_true", help="Skip Step A")
    ap.add_argument("--skip-grading", action="store_true", help="Skip Step B")
    ap.add_argument("--skip-correlation", action="store_true", help="Skip Step C")
    ap.add_argument("--skip-prediction", action="store_true", help="Skip Step D")
    return ap.parse_args()


def prompt_if_missing(args):
    name = args.name or input("Participant name (e.g., MiguelNg): ").strip()
    video = Path(args.video) if args.video else None
    transcript = Path(args.transcript) if args.transcript else None
    summary = Path(args.summary) if args.summary else None
    group = (
        args.group
        or input("Group [Neutral/Opposing/Similar] (optional): ").strip()
        or None
    )
    return name, video, transcript, summary, group


def main():
    args = parse_args()
    name, video, transcript, summary, group = prompt_if_missing(args)

    if not args.skip_ingestion:
        step_a_ingestion(name, video)
    else:
        print("[skip] Step A")

    if not args.skip_grading:
        step_b_grading(name, transcript, summary, group)
    else:
        print("[skip] Step B")

    if not args.skip_correlation:
        step_c_correlation()
    else:
        print("[skip] Step C")

    if not args.skip_prediction:
        step_d_prediction(name)
    else:
        print("[skip] Step D")

    print("\nPipeline complete. Outputs are available under 'outputs/'.")


if __name__ == "__main__":
    main()
