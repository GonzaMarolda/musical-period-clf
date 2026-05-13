import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import librosa
import pretty_midi


# --- Constants ---------------------------------------------------------------

REQUIRED_MAESTRO_COLUMNS = [
    "canonical_composer",
    "canonical_title",
    "split",
    "year",
    "midi_filename",
    "audio_filename",
    "duration",
]

REQUIRED_MAP_COLUMNS = ["canonical_composer", "period", "period_id"]

CLEAN_COLUMNS = [
    "track_id",
    "canonical_composer",
    "canonical_title",
    "year",
    "duration",
    "midi_filename",
    "audio_filename",
    "midi_path",
    "audio_path",
    "official_split",
    "composition_id",
]

LABELED_COLUMNS = CLEAN_COLUMNS + ["period", "period_id"]

N_MFCC = 13
N_CHROMA = 12

# --- PATHS ---------------------------------------------------------------------

MAESTRO_DIR = Path("data/raw/maestro-v3.0.0")
COMPOSER_MAP_DIR = Path("data/map/composer_period_map.csv")
INTERIM_DIR = Path("data/interm")
PROCESSED_DIR = Path("data/processed")

# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MAESTRO v3.0.0 into a feature table."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate for audio loading.",
    )
    parser.add_argument(
        "--audio-window-seconds",
        type=float,
        default=60.0,
        help="Length of the centered audio window used for feature extraction.",
    )
    return parser.parse_args()


# --- Helpers -----------------------------------------------------------------

def _slugify(text: str) -> str:
    """Lowercase, ASCII-ish, underscore-separated identifier."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _audio_feature_names() -> list[str]:
    base = [
        "duration_sec",
        "tempo_bpm",
        "rms_mean",
        "rms_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
    ]
    for i in range(1, N_MFCC + 1):
        base.append(f"mfcc_{i}_mean")
        base.append(f"mfcc_{i}_std")
    for i in range(1, N_CHROMA + 1):
        base.append(f"chroma_{i}_mean")
    return base


def _midi_feature_names() -> list[str]:
    return [
        "midi_note_count",
        "midi_notes_per_second",
        "midi_pitch_mean",
        "midi_pitch_std",
        "midi_pitch_min",
        "midi_pitch_max",
        "midi_pitch_range",
        "midi_velocity_mean",
        "midi_velocity_std",
        "midi_duration_mean",
        "midi_duration_std",
        "midi_polyphony_mean",
    ]


def _nan_audio_features() -> dict:
    return {name: np.nan for name in _audio_feature_names()}


def _nan_midi_features() -> dict:
    return {name: np.nan for name in _midi_feature_names()}


# --- Metadata loaders --------------------------------------------------------

def load_maestro_metadata(maestro_dir: Path) -> pd.DataFrame:
    """Load and validate the original MAESTRO metadata CSV."""
    csv_path = maestro_dir / "maestro-v3.0.0.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MAESTRO metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_MAESTRO_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"MAESTRO metadata is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def build_clean_metadata(meta: pd.DataFrame, maestro_dir: Path) -> pd.DataFrame:
    """Project the raw metadata into the clean schema with stable IDs and paths."""
    df = meta.reset_index(drop=True).copy()

    # Stable per-row identifier.
    df["track_id"] = [f"track_{i + 1:06d}" for i in range(len(df))]

    # Build full paths relative to the MAESTRO root.
    df["midi_path"] = df["midi_filename"].apply(lambda p: str(maestro_dir / p))
    df["audio_path"] = df["audio_filename"].apply(lambda p: str(maestro_dir / p))

    # Rename split -> official_split.
    df["official_split"] = df["split"]

    # Composition fingerprint (same piece across performances).
    df["composition_id"] = df.apply(
        lambda r: f"{_slugify(r['canonical_composer'])}__{_slugify(r['canonical_title'])}",
        axis=1,
    )

    return df[CLEAN_COLUMNS].copy()


def load_composer_map(map_path: Path) -> pd.DataFrame:
    """Load and validate the composer -> period mapping."""
    if not map_path.exists():
        raise FileNotFoundError(f"Composer period map not found: {map_path}")

    df = pd.read_csv(map_path)
    missing = [c for c in REQUIRED_MAP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Composer map is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df[REQUIRED_MAP_COLUMNS].copy()


def build_labeled_metadata(
    clean_df: pd.DataFrame, composer_map: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join clean metadata with the composer map to drop unmapped composers."""
    merged = clean_df.merge(composer_map, on="canonical_composer", how="inner")
    return merged[LABELED_COLUMNS].reset_index(drop=True)


# --- Feature extraction ------------------------------------------------------

def extract_audio_features(
    audio_path: Path, sample_rate: int, window_seconds: float
) -> dict:
    """Extract a fixed-length set of audio features from a centered window."""
    # Use the WAV header to get duration without decoding the whole file.
    full_duration = float(librosa.get_duration(path=str(audio_path)))

    if full_duration <= window_seconds:
        offset = 0.0
        target_duration = full_duration
    else:
        offset = max(0.0, (full_duration - window_seconds) / 2.0)
        target_duration = window_seconds

    y, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
        offset=offset,
        duration=target_duration,
    )
    if y.size == 0:
        raise ValueError("Empty audio after loading.")

    features: dict = {}
    features["duration_sec"] = float(librosa.get_duration(y=y, sr=sr))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo_bpm"] = float(np.atleast_1d(tempo)[0])

    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(sc))
    features["spectral_centroid_std"] = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(sb))
    features["spectral_bandwidth_std"] = float(np.std(sb))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    features["zero_crossing_rate_std"] = float(np.std(zcr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfcc[i]))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(N_CHROMA):
        features[f"chroma_{i + 1}_mean"] = float(np.mean(chroma[i]))

    return features


def extract_midi_features(midi_path: Path) -> dict:
    """Extract aggregate MIDI features from all instruments combined."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments for n in inst.notes]

    if len(notes) == 0:
        raise ValueError("No notes in MIDI file.")

    pitches = np.array([n.pitch for n in notes], dtype=float)
    velocities = np.array([n.velocity for n in notes], dtype=float)
    durations = np.array([n.end - n.start for n in notes], dtype=float)

    end_time = float(max(n.end for n in notes))
    notes_per_second = len(notes) / end_time if end_time > 0 else 0.0

    # Approximate polyphony: mean active-note count on a 100 ms grid,
    # only counting cells where at least one note is sounding.
    grid_step = 0.1
    if end_time > 0:
        n_cells = int(np.ceil(end_time / grid_step))
        active = np.zeros(n_cells, dtype=np.int32)
        for n in notes:
            i0 = int(n.start / grid_step)
            i1 = int(n.end / grid_step)
            if i0 < 0:
                i0 = 0
            if i1 >= n_cells:
                i1 = n_cells - 1
            if i1 >= i0:
                active[i0:i1 + 1] += 1
        sounding = active[active > 0]
        polyphony_mean = float(np.mean(sounding)) if sounding.size > 0 else 0.0
    else:
        polyphony_mean = 0.0

    return {
        "midi_note_count": int(len(notes)),
        "midi_notes_per_second": float(notes_per_second),
        "midi_pitch_mean": float(np.mean(pitches)),
        "midi_pitch_std": float(np.std(pitches)),
        "midi_pitch_min": float(np.min(pitches)),
        "midi_pitch_max": float(np.max(pitches)),
        "midi_pitch_range": float(np.max(pitches) - np.min(pitches)),
        "midi_velocity_mean": float(np.mean(velocities)),
        "midi_velocity_std": float(np.std(velocities)),
        "midi_duration_mean": float(np.mean(durations)),
        "midi_duration_std": float(np.std(durations)),
        "midi_polyphony_mean": polyphony_mean,
    }


def extract_features_for_row(
    row: pd.Series, sample_rate: int, window_seconds: float
) -> dict:
    """Run audio + MIDI extraction for one row, tagging which side failed."""
    audio_ok = True
    midi_ok = True

    try:
        audio_features = extract_audio_features(
            Path(row["audio_path"]), sample_rate, window_seconds
        )
    except Exception:
        audio_features = _nan_audio_features()
        audio_ok = False

    try:
        midi_features = extract_midi_features(Path(row["midi_path"]))
    except Exception:
        midi_features = _nan_midi_features()
        midi_ok = False

    if audio_ok and midi_ok:
        status = "ok"
    elif not audio_ok and midi_ok:
        status = "audio_error"
    elif audio_ok and not midi_ok:
        status = "midi_error"
    else:
        status = "audio_midi_error"

    return {**audio_features, **midi_features, "feature_status": status}


def build_final_dataset(
    labeled_df: pd.DataFrame, sample_rate: int, window_seconds: float
) -> pd.DataFrame:
    """Run feature extraction over every labeled row and join with metadata."""
    total = len(labeled_df)
    feature_rows: list[dict] = []

    for idx, (_, row) in enumerate(labeled_df.iterrows(), start=1):
        feats = extract_features_for_row(row, sample_rate, window_seconds)
        feature_rows.append(feats)

        # Lightweight progress signal for long runs.
        if idx % 50 == 0 or idx == total:
            print(f"  processed {idx}/{total}", flush=True)

    features_df = pd.DataFrame(feature_rows)
    final_df = pd.concat(
        [labeled_df.reset_index(drop=True), features_df.reset_index(drop=True)],
        axis=1,
    )
    return final_df


# --- Outputs -----------------------------------------------------------------

def save_feature_columns(processed_dir: Path) -> Path:
    """Persist the list of training feature columns as JSON."""
    columns = _audio_feature_names() + _midi_feature_names()
    out_path = processed_dir / "feature_columns.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2)
    return out_path


def print_summary(
    meta_df: pd.DataFrame, labeled_df: pd.DataFrame, final_df: pd.DataFrame
) -> None:
    n_total = len(meta_df)
    n_labeled = len(labeled_df)
    n_processed = len(final_df)
    n_ok = int((final_df["feature_status"] == "ok").sum())
    n_err = n_processed - n_ok

    print("=" * 50)
    print("Resumen de preprocesado")
    print("=" * 50)
    print(f"Filas metadata original   : {n_total}")
    print(f"Filas metadata_labeled    : {n_labeled}")
    print(f"Filas procesadas          : {n_processed}")
    print(f"Filas feature_status ok   : {n_ok}")
    print(f"Filas con errores         : {n_err}")
    print("Distribución de period:")
    if "period" in final_df.columns and len(final_df) > 0:
        counts = final_df["period"].value_counts(dropna=False)
        for period, count in counts.items():
            print(f"  {period}: {count}")
    else:
        print("  (sin datos)")


# --- Entry point -------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Silence noisy warnings from librosa/audioread on edge cases.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Reading MAESTRO metadata...", flush=True)
    meta = load_maestro_metadata(MAESTRO_DIR)

    print("[2/4] Building clean metadata...", flush=True)
    clean_df = build_clean_metadata(meta, MAESTRO_DIR)
    clean_path = INTERIM_DIR / "clean.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"      wrote {clean_path} ({len(clean_df)} rows)", flush=True)

    print("[3/4] Building labeled metadata...", flush=True)
    composer_map = load_composer_map(COMPOSER_MAP_DIR)
    labeled_df = build_labeled_metadata(clean_df, composer_map)
    labeled_path = INTERIM_DIR / "labeled.csv"
    labeled_df.to_csv(labeled_path, index=False)
    print(f"      wrote {labeled_path} ({len(labeled_df)} rows)", flush=True)

    print("[4/4] Extracting audio + MIDI features...", flush=True)
    final_df = build_final_dataset(
        labeled_df,
        sample_rate=args.sample_rate,
        window_seconds=args.audio_window_seconds,
    )
    features_path = PROCESSED_DIR / "maestro_features.csv"
    final_df.to_csv(features_path, index=False)
    print(f"      wrote {features_path} ({len(final_df)} rows)", flush=True)

    columns_path = save_feature_columns(PROCESSED_DIR)
    print(f"      wrote {columns_path}", flush=True)

    print_summary(meta, labeled_df, final_df)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
