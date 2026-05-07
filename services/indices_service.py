from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INDICES_DIR = BASE_DIR / "data" / "Indices"
CSV_PATH = BASE_DIR / "data" / "indices_timeseries.csv"

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


def _read_indices_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=["date", "roi", "value", "index"])

    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["index"] = df["index"].astype(str)
    df["roi"] = df["roi"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "roi", "index", "value"])
    return df


def list_indices() -> list[str]:
    df = _read_indices_csv()

    if not df.empty:
        return sorted(df["index"].dropna().unique().tolist())

    return [idx for idx in VALID_INDICES if (INDICES_DIR / idx).exists()]


def load_index_array(index_name: str, roi: str) -> np.ndarray:
    path = INDICES_DIR / index_name / f"{roi}.npy"

    if not path.exists():
        raise FileNotFoundError(f"Fișierul nu există: {path}")

    arr = np.load(path)

    if arr.ndim != 3:
        raise ValueError(f"Array-ul trebuie să fie 3D: H x W x T. Shape primit: {arr.shape}")

    return arr


def array_to_mean_series(arr: np.ndarray, start_date: str = "2017-01-01") -> pd.Series:
    t_len = arr.shape[2]
    values = []

    for t in range(t_len):
        frame = arr[:, :, t].astype(float)
        frame = np.where(np.isfinite(frame), frame, np.nan)
        values.append(float(np.nanmean(frame)))

    dates = pd.date_range(start=start_date, periods=t_len, freq="MS")
    return pd.Series(values, index=dates)


def load_index_series(index_name: str, roi: str, start_date: str = "2017-01-01") -> pd.Series:
    df = _read_indices_csv()

    if not df.empty:
        sub = df[(df["index"] == index_name) & (df["roi"] == roi)].copy()

        if sub.empty:
            return pd.Series(dtype=float)

        return sub.sort_values("date").set_index("date")["value"]

    arr = load_index_array(index_name, roi)
    return array_to_mean_series(arr, start_date=start_date)


def load_index_dataframe(index_name: str, start_date: str = "2017-01-01") -> pd.DataFrame:
    df = _read_indices_csv()

    if not df.empty:
        return df[df["index"] == index_name].copy()

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
    df = _read_indices_csv()

    if not df.empty:
        return df.copy()

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
    df = load_all_indices_dataframe(start_date=start_date)
    return df[df["roi"] == roi].copy()


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