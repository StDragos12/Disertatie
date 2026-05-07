import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller


def prepare_monthly_series(sub: pd.DataFrame, value_col: str = "ndvi") -> pd.Series:
    s = sub.sort_values("date").set_index("date")[value_col].copy()
    s = s.asfreq("MS")
    s = s.interpolate(method="time").ffill().bfill()
    return s


def stl_series(series: pd.Series, period: int = 12):
    s = series.sort_index().copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Seria STL trebuie să aibă DatetimeIndex.")
    s = s.asfreq("MS")
    s = s.interpolate(method="time").ffill().bfill()
    if len(s.dropna()) < 24:
        raise ValueError("Sunt necesare cel puțin 24 observații lunare pentru STL.")
    return STL(s, period=period, robust=True).fit()


def stationarity_metrics_from_series(series: pd.Series) -> dict:
    s = series.dropna().copy()
    if len(s) < 12:
        return {
            "adf_stat": None,
            "p_value": None,
            "lags_used": None,
            "n_obs": int(len(s)),
            "stationary": "Date insuficiente",
            "series": series,
        }

    result = adfuller(s, autolag="AIC")
    adf_stat = float(result[0])
    p_value = float(result[1])
    lags_used = int(result[2])
    n_obs = int(result[3])
    interpretation = "Staționară" if p_value < 0.05 else "Nestaționară"

    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "lags_used": lags_used,
        "n_obs": n_obs,
        "stationary": interpretation,
        "series": series,
    }


def count_anomalies_in_series(series: pd.Series, period: int = 12) -> int:
    try:
        res = stl_series(series, period=period)
        resid = res.resid.dropna()
        std = resid.std(ddof=0)
        if std == 0 or pd.isna(std):
            return 0
        z = (resid - resid.mean()) / std
        return int((abs(z) >= 2).sum())
    except Exception:
        return 0


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float(np.mean(np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1])))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float(np.sqrt(np.mean((aligned.iloc[:, 0] - aligned.iloc[:, 1]) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if aligned.empty:
        return np.nan
    y = aligned.iloc[:, 0]
    p = aligned.iloc[:, 1]
    valid = y != 0
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y[valid] - p[valid]) / y[valid])) * 100)


def extract_features(series: pd.Series) -> dict:
    station = stationarity_metrics_from_series(series)
    anomalies = count_anomalies_in_series(series, period=12)

    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "amplitude": float(series.max() - series.min()),
        "adf_pvalue": None if station["p_value"] is None else float(station["p_value"]),
        "stationary": station["stationary"],
        "anomalies": int(anomalies),
    }


def classify_series_features(features: dict) -> str:
    stationary = features["stationary"] == "Staționară"
    anomalies = features["anomalies"]
    amplitude = features["amplitude"]

    if stationary and anomalies <= 4 and amplitude < 5:
        return "Stabilă"

    if (not stationary) and amplitude >= 2:
        return "Trending"

    return "Mixtă"


def _zscore_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    std = arr.std(ddof=0)
    if std == 0 or np.isnan(std):
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def dtw_distance(s1: pd.Series, s2: pd.Series, normalize: bool = True) -> float:
    a = s1.dropna().values.astype(float)
    b = s2.dropna().values.astype(float)

    if normalize:
        a = _zscore_array(a)
        b = _zscore_array(b)

    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    return float(dp[n, m])


def pairwise_dtw_matrix(series_items: list[tuple[str, pd.Series]], normalize: bool = True) -> pd.DataFrame:
    names = [name for name, _ in series_items]
    matrix = np.zeros((len(series_items), len(series_items)), dtype=float)

    for i, (_, s1) in enumerate(series_items):
        for j, (_, s2) in enumerate(series_items):
            if i <= j:
                d = dtw_distance(s1, s2, normalize=normalize)
                matrix[i, j] = d
                matrix[j, i] = d

    return pd.DataFrame(matrix, index=names, columns=names)