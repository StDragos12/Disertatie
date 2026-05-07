from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INDICES_DIR = BASE_DIR / "data" / "Indices"

VALID_INDICES = ["AVI", "EVI", "GCI", "GNDVI", "MSI", "NDVI", "SAVI"]
VALID_ROIS = ["roi1", "roi2"]

INDEX_DESCRIPTIONS = {
    "NDVI": "Normalized Difference Vegetation Index – indică densitatea și sănătatea vegetației.",
    "EVI": "Enhanced Vegetation Index – mai robust pentru vegetație densă și efecte atmosferice.",
    "SAVI": "Soil Adjusted Vegetation Index – reduce influența solului în zone cu vegetație rară.",
    "GNDVI": "Green NDVI – sensibil la conținutul de clorofilă.",
    "GCI": "Green Chlorophyll Index – utilizat pentru estimarea clorofilei.",
    "MSI": "Moisture Stress Index – indică stresul hidric al vegetației.",
    "AVI": "Advanced Vegetation Index – evidențiază densitatea vegetației.",
}


def list_indices() -> list[str]:
    available = []
    for index_name in VALID_INDICES:
        if (INDICES_DIR / index_name).exists():
            available.append(index_name)
    return available


def load_index_array(index_name: str, roi: str) -> np.ndarray:
    if index_name not in VALID_INDICES:
        raise ValueError(f"Indice invalid: {index_name}")

    if roi not in VALID_ROIS:
        raise ValueError(f"ROI invalid: {roi}")

    path = INDICES_DIR / index_name / f"{roi}.npy"

    if not path.exists():
        raise FileNotFoundError(f"Fișierul nu există: {path}")

    arr = np.load(path)

    if arr.ndim != 3:
        raise ValueError(f"Array-ul trebuie să fie 3D: H x W x T. Shape primit: {arr.shape}")

    return arr


def array_to_mean_series(arr: np.ndarray, start_date: str = "2017-01-01") -> pd.Series:
    """
    Transformă un stack H x W x T într-o serie temporală prin media pixelilor pentru fiecare moment T.
    """
    t_len = arr.shape[2]

    values = []
    for t in range(t_len):
        frame = arr[:, :, t].astype(float)

        # eliminăm valori invalide
        frame = np.where(np.isfinite(frame), frame, np.nan)

        values.append(float(np.nanmean(frame)))

    dates = pd.date_range(start=start_date, periods=t_len, freq="MS")
    return pd.Series(values, index=dates)


def load_index_series(index_name: str, roi: str, start_date: str = "2017-01-01") -> pd.Series:
    arr = load_index_array(index_name, roi)
    return array_to_mean_series(arr, start_date=start_date)


def load_index_dataframe(index_name: str, start_date: str = "2017-01-01") -> pd.DataFrame:
    rows = []

    for roi in VALID_ROIS:
        series = load_index_series(index_name, roi, start_date=start_date)

        for date, value in series.items():
            rows.append({
                "date": date,
                "roi": roi,
                "value": value,
                "index": index_name,
            })

    return pd.DataFrame(rows)


def load_all_indices_dataframe(start_date: str = "2017-01-01") -> pd.DataFrame:
    frames = []

    for index_name in list_indices():
        try:
            frames.append(load_index_dataframe(index_name, start_date=start_date))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["date", "roi", "value", "index"])

    return pd.concat(frames, ignore_index=True)

def build_cross_index_dataframe(roi: str = "roi1", start_date: str = "2017-01-01") -> pd.DataFrame:
    rows = []

    for index_name in list_indices():
        try:
            series = load_index_series(index_name, roi, start_date=start_date)

            for date, value in series.items():
                rows.append({
                    "date": date,
                    "index": index_name,
                    "value": value,
                    "roi": roi,
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


def build_indices_wide_dataframe(roi: str = "roi1", start_date: str = "2017-01-01") -> pd.DataFrame:
    df = build_cross_index_dataframe(roi=roi, start_date=start_date)

    if df.empty:
        return pd.DataFrame()

    wide = df.pivot_table(
        index="date",
        columns="index",
        values="value",
        aggfunc="mean",
    ).sort_index()

    return wide.dropna(axis=1, how="all")