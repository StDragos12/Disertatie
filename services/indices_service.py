import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from google.cloud import storage
except Exception:
    storage = None

from services.dataset_service import (
    DEMO_DATASET_ID,
    normalize_dataset_id,
    load_dataset_dataframe,
    as_roi_level_dataframe,
)


DATA_DIR = Path("data")
INDICES_DIR = DATA_DIR / "Indices"
INDICES_CSV_PATH = DATA_DIR / "indices_timeseries.csv"


INDEX_DESCRIPTIONS = {
    "NDVI": "Normalized Difference Vegetation Index – indică densitatea și sănătatea vegetației.",
    "NDMI": "Normalized Difference Moisture Index – evidențiază conținutul de umiditate al vegetației.",
    "SAVI": "Soil Adjusted Vegetation Index – reduce influența solului asupra estimării vegetației.",
    "AVI": "Advanced Vegetation Index – evidențiază variațiile vegetației în zone cu acoperire vegetală diferită.",
    "EVI": "Enhanced Vegetation Index – îmbunătățește sensibilitatea în zone cu vegetație densă.",
    "GNDVI": "Green Normalized Difference Vegetation Index – folosește banda verde pentru analiza vigorii vegetației.",
    "GCI": "Green Chlorophyll Index – indicator asociat cu conținutul de clorofilă.",
    "MSI": "Moisture Stress Index – indicator asociat cu stresul hidric al vegetației.",
}


def _load_demo_indices_csv() -> pd.DataFrame:
    if not INDICES_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Fișierul CSV nu există: {INDICES_CSV_PATH}"
        )

    df = pd.read_csv(INDICES_CSV_PATH)
    return _normalize_indices_dataframe(df, source=str(INDICES_CSV_PATH))


def _normalize_indices_dataframe(df: pd.DataFrame, source: str = "CSV") -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required_columns = {"date", "roi", "index", "value"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Lipsesc coloanele necesare din {source}: {missing_columns}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["roi"] = df["roi"].astype(str).str.strip().str.lower()
    df["index"] = df["index"].astype(str).str.strip().str.upper()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "roi", "index", "value"])
    df = df[df["roi"] != ""]
    df = df[df["index"] != ""]

    return df.sort_values(["index", "roi", "date"])


def load_indices_dataframe(dataset_id: str = DEMO_DATASET_ID) -> pd.DataFrame:
    dataset_id = normalize_dataset_id(dataset_id)

    if dataset_id == DEMO_DATASET_ID:
        return _load_demo_indices_csv()

    # Dataseturile pixel-level sunt agregate automat la nivel ROI/date/index
    # pentru modulele de analiză spectrală, Cross-Index, validare și forecast.
    return as_roi_level_dataframe(load_dataset_dataframe(dataset_id))


def list_indices(dataset_id: str = DEMO_DATASET_ID):
    df = load_indices_dataframe(dataset_id)

    return sorted(
        df["index"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )


def load_index_dataframe(index_name: str, dataset_id: str = DEMO_DATASET_ID) -> pd.DataFrame:
    selected_index = index_name.upper()

    df = load_indices_dataframe(dataset_id)

    df = df[df["index"] == selected_index].copy()

    if df.empty:
        raise ValueError(
            f"Nu există date pentru indicele {selected_index}."
        )

    return df.sort_values(["roi", "date"])


def build_indices_wide_dataframe(roi: str = "roi1", dataset_id: str = DEMO_DATASET_ID) -> pd.DataFrame:
    selected_roi = roi.lower()

    df = load_indices_dataframe(dataset_id)

    df = df[df["roi"] == selected_roi].copy()

    if df.empty:
        return pd.DataFrame()

    wide_df = (
        df
        .pivot_table(
            index="date",
            columns="index",
            values="value",
            aggfunc="mean"
        )
        .sort_index()
    )

    return wide_df


def smooth_series(series: pd.Series, window: int = 3) -> pd.Series:
    if series.empty:
        return series

    return (
        series
        .rolling(
            window=window,
            min_periods=1,
            center=True
        )
        .mean()
    )


def _download_index_array_from_gcs(index_name: str, roi: str) -> Path:
    bucket_name = os.getenv("GCS_BUCKET_NAME")

    if not bucket_name:
        raise FileNotFoundError(
            "Fișierul .npy nu există local și variabila GCS_BUCKET_NAME nu este setată."
        )

    if storage is None:
        raise ImportError(
            "google-cloud-storage nu este instalat. Adaugă google-cloud-storage în requirements.txt."
        )

    index_name = index_name.upper()
    roi = roi.lower()

    tmp_dir = Path("/tmp") / "Indices" / index_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = tmp_dir / f"{roi}.npy"

    if tmp_path.exists():
        return tmp_path

    blob_name = f"data/Indices/{index_name}/{roi}.npy"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(
            f"Fișierul nu există în Cloud Storage: gs://{bucket_name}/{blob_name}"
        )

    blob.download_to_filename(str(tmp_path))

    return tmp_path


def get_index_array_path(index_name: str, roi: str) -> Path:
    index_name = index_name.upper()
    roi = roi.lower()

    local_path = INDICES_DIR / index_name / f"{roi}.npy"

    if local_path.exists():
        return local_path

    return _download_index_array_from_gcs(index_name, roi)


def load_index_array(index_name: str, roi: str) -> np.ndarray:
    path = get_index_array_path(index_name, roi)

    return np.load(path)
