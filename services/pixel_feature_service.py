from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

BASE_DIR = Path(__file__).resolve().parent.parent
INDICES_DIR = BASE_DIR / "data" / "Indices"


def extract_pixel_features(
    index_name: str = "NDVI",
    roi: str = "roi1"
) -> pd.DataFrame:

    path = INDICES_DIR / index_name / f"{roi}.npy"

    arr = np.load(path)

    h, w, t = arr.shape

    rows = []

    for i in range(h):
        for j in range(w):

            series = arr[i, j, :].astype(float)

            series = np.where(
                np.isfinite(series),
                series,
                np.nan
            )

            if np.all(np.isnan(series)):
                continue

            mean_val = np.nanmean(series)
            std_val = np.nanstd(series)

            min_val = np.nanmin(series)
            max_val = np.nanmax(series)

            amplitude = max_val - min_val

            x = np.arange(len(series))

            mask = ~np.isnan(series)

            if np.sum(mask) > 2:
                slope = linregress(
                    x[mask],
                    series[mask]
                ).slope
            else:
                slope = 0.0

            rows.append({
                "x": i,
                "y": j,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "amplitude": amplitude,
                "slope": slope,
            })

    return pd.DataFrame(rows)