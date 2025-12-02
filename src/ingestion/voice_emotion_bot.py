import os
from typing import Optional

import numpy as np
import pandas as pd
import librosa


class VoiceEmotionBot:
    """
    Lightweight voice emotion backend.

    Provides a rule-based fallback using acoustic features (pitch, energy, speech rate)
    to estimate arousal/valence/intensity and simple emotion probabilities.

    This is intended to be dependency-light and CPU-only. If you have an ML model,
    you can wire it by setting use_ml_model=True and implementing the relevant path.
    """

    def __init__(self, sample_rate: int = 22050, use_ml_model: bool = False):
        self.sample_rate = int(sample_rate)
        self.use_ml_model = bool(use_ml_model)

    def _analyze_segment(self, y: np.ndarray) -> dict:
        # Energy
        energy = float(np.sqrt(np.mean(y**2)))
        # Pitch via librosa piptrack (rough estimate)
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        pitches, mags = librosa.piptrack(S=S, sr=self.sample_rate, hop_length=256)
        pitch_vals = pitches[mags > np.median(mags) if mags.size else slice(None)]
        pitch_hz = float(np.nanmean(pitch_vals)) if pitch_vals.size else 0.0
        # Speech rate proxy: zero-crossing rate
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Normalize features to rough 0..1 ranges
        pitch_norm = np.clip(pitch_hz / 300.0, 0.0, 1.0)  # 300 Hz cap
        energy_norm = np.clip(energy * 10.0, 0.0, 1.0)  # scale RMS
        zcr_norm = np.clip(zcr * 5.0, 0.0, 1.0)

        # Rule-based estimates
        arousal = float(np.clip(0.6 * energy_norm + 0.4 * zcr_norm, 0.0, 1.0))
        valence = float(
            np.clip(
                0.5 * pitch_norm - 0.2 * zcr_norm + 0.5 * (1 - energy_norm), -1.0, 1.0
            )
        )
        intensity = float(np.clip(energy_norm, 0.0, 1.0))

        # Basic emotion probabilities (softmax-ish without strict normalization)
        happy = float(
            np.clip(
                0.5 * pitch_norm + 0.3 * (1 - zcr_norm) + 0.2 * (1 - energy_norm),
                0.0,
                1.0,
            )
        )
        angry = float(
            np.clip(0.6 * energy_norm + 0.3 * zcr_norm - 0.1 * pitch_norm, 0.0, 1.0)
        )
        sad = float(np.clip(0.5 * (1 - pitch_norm) + 0.3 * (1 - energy_norm), 0.0, 1.0))

        return {
            "voice_arousal": arousal,
            "voice_valence": valence,
            "voice_intensity": intensity,
            "voice_happy": happy,
            "voice_angry": angry,
            "voice_sad": sad,
            "pitch_mean": pitch_hz,
            "volume_mean": energy,
            "speech_rate": zcr,
        }

    def process_audio_file(
        self, audio_path: str, segment_duration: float = 1.0
    ) -> pd.DataFrame:
        if not os.path.exists(audio_path):
            return pd.DataFrame()

        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception:
            return pd.DataFrame()

        seg_len = max(int(segment_duration * self.sample_rate), 1)
        n_segments = max(int(np.ceil(len(y) / seg_len)), 1)

        rows = []
        for i in range(n_segments):
            start = i * seg_len
            end = min((i + 1) * seg_len, len(y))
            seg = y[start:end]
            if seg.size == 0:
                continue
            feats = self._analyze_segment(seg)
            feats["time_seconds"] = float(start / self.sample_rate)
            rows.append(feats)

        return pd.DataFrame(rows)
