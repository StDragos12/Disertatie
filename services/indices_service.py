import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from google.cloud import storage
except Exception:
    storage = None


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


def _load_indices_csv() -> pd.DataFrame:
    if not INDICES_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Fișierul CSV nu există: {INDICES_CSV_PATH}"
        )

    df = pd.read_csv(INDICES_CSV_PATH)

    required_columns = {"date", "roi", "index", "value"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Lipsesc coloanele necesare din {INDICES_CSV_PATH}: {missing_columns}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date", "roi", "index", "value"]
    )

    df["index"] = df["index"].astype(str).str.upper()
    df["roi"] = df["roi"].astype(str).str.lower()
    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["value"]
    )

    return df.sort_values(
        ["index", "roi", "date"]
    )


def list_indices():
    df = _load_indices_csv()

    return sorted(
        df["index"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )


def load_index_dataframe(index_name: str) -> pd.DataFrame:
    selected_index = index_name.upper()

    df = _load_indices_csv()

    df = df[
        df["index"] == selected_index
    ].copy()

    if df.empty:
        raise ValueError(
            f"Nu există date pentru indicele {selected_index}."
        )

    return df.sort_values(
        ["roi", "date"]
    )


def build_indices_wide_dataframe(roi: str = "roi1") -> pd.DataFrame:
    selected_roi = roi.lower()

    df = _load_indices_csv()

    df = df[
        df["roi"] == selected_roi
    ].copy()

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
    tmp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

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

    blob.download_to_filename(
        str(tmp_path)
    )

    return tmp_path


def get_index_array_path(index_name: str, roi: str) -> Path:
    index_name = index_name.upper()
    roi = roi.lower()

    local_path = INDICES_DIR / index_name / f"{roi}.npy"

    if local_path.exists():
        return local_path

    return _download_index_array_from_gcs(
        index_name,
        roi
    )


def load_index_array(index_name: str, roi: str) -> np.ndarray:
    path = get_index_array_path(
        index_name,
        roi
    )

    return np.load(path)