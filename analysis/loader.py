import os
from typing import Dict, List, Tuple
import pandas as pd


def find_fusion_files(results_dir: str) -> List[str]:
    """Recursively find fusion CSVs under results directory.

    Matches filenames like `Final*_ml_fusion.csv`.
    """
    files = []
    for root, _, filenames in os.walk(results_dir):
        for fn in filenames:
            if fn.startswith("Final") and fn.endswith("_ml_fusion.csv"):
                files.append(os.path.join(root, fn))
    return sorted(files)


def load_participant_csv(path: str) -> pd.DataFrame:
    """Load a participant fusion CSV into a DataFrame.

    Tries to parse `timestamp` column if present.
    """
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            pass
    return df


def group_from_path(path: str, results_dir: str) -> Tuple[str, str]:
    """Return (group, participant_id) extracted from a file path.

    Expects path like `.../results/<Group>/<Participant>/Final...csv`.
    """
    rel = os.path.relpath(path, results_dir)
    parts = rel.split(os.path.sep)
    if len(parts) >= 3:
        group = parts[0]
        participant = parts[1]
    elif len(parts) >= 2:
        group = parts[0]
        participant = parts[1]
    else:
        group = "unknown"
        participant = os.path.splitext(os.path.basename(path))[0]
    return group, participant


def load_all_groups(results_dir: str = "results") -> Dict[str, List[Tuple[str, pd.DataFrame]]]:
    """Load all fusion CSVs organized by group.

    Returns mapping group -> list of (participant_id, df)
    """
    files = find_fusion_files(results_dir)
    groups = {}
    for f in files:
        group, participant = group_from_path(f, results_dir)
        df = load_participant_csv(f)
        groups.setdefault(group, []).append((participant, df))
    return groups
