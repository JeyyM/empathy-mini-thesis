from typing import List
import numpy as np
import pandas as pd


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    # common numeric columns to consider
    candidates = [
        "facial_arousal",
        "facial_valence",
        "facial_intensity",
        "voice_arousal",
        "voice_valence",
        "voice_intensity",
        "combined_arousal",
        "combined_valence",
        "combined_intensity",
        "fused_arousal",
        "fused_valence",
        "fused_intensity",
    ]
    return [c for c in candidates if c in df.columns]


def resample_numeric_series(x: np.ndarray, y: np.ndarray, n: int = 100) -> np.ndarray:
    """Resample (x,y) to n evenly spaced samples along x via linear interp."""
    if len(x) == 0:
        return np.full(n, np.nan)
    # ensure increasing x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    new_x = np.linspace(x_sorted[0], x_sorted[-1], n)
    return np.interp(new_x, x_sorted, y_sorted)


def resample_participant(df: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
    """Resample participant time series to fixed-length (n_points).

    Numeric features are interpolated. Categorical `fused_dominant_emotion`
    is resampled by taking the most frequent emotion in each temporal bin.
    Returns DataFrame with index 0..n_points-1 and columns for numeric + `dominant_emotion`.
    """
    if "time_seconds" in df.columns:
        t = df["time_seconds"].to_numpy()
    elif "timestamp" in df.columns:
        # use seconds from start
        t = (pd.to_datetime(df["timestamp"]) - pd.to_datetime(df["timestamp"]).iloc[0]).dt.total_seconds().to_numpy()
    else:
        # fallback to row index
        t = np.arange(len(df))

    numeric = _numeric_cols(df)
    out = {}
    for col in numeric:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        out[col] = resample_numeric_series(t, y, n=n_points)

    # dominant emotion column detection
    dom_col = None
    for candidate in ("fused_dominant_emotion", "fused_dominant", "dominant_emotion", "fused_dominant_emotion"):
        if candidate in df.columns:
            dom_col = candidate
            break

    if dom_col is not None:
        # create bins and pick most frequent label per bin
        labels = df[dom_col].astype(str).to_numpy()
        # edges
        bins = np.linspace(t[0], t[-1] if len(t) else 0, n_points + 1)
        binned = []
        for i in range(n_points):
            mask = (t >= bins[i]) & (t <= bins[i + 1])
            if mask.any():
                vals, counts = np.unique(labels[mask], return_counts=True)
                binned.append(vals[np.argmax(counts)])
            else:
                binned.append("")
        out["dominant_emotion"] = np.array(binned)

    result = pd.DataFrame(out)
    result.index.name = "segment"
    return result
