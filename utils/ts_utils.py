import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress


def prepare_monthly_series(sub):

    df = sub.copy()

    if "date" not in df.columns:
        raise ValueError("Lipsește coloana date")

    value_col = None

    for col in ["ndvi", "NDVI", "value"]:
        if col in df.columns:
            value_col = col
            break

    if value_col is None:
        raise ValueError("Nu există coloană NDVI/value")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    # IMPORTANT
    # agregăm duplicatele pe aceeași lună

    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby("month")[value_col]
        .mean()
        .reset_index()
    )

    monthly["date"] = monthly["month"].dt.to_timestamp()

    series = pd.Series(
        monthly[value_col].values,
        index=monthly["date"]
    )

    series = series.asfreq("MS")

    return series


def stl_series(series: pd.Series, period: int = 12):
    s = series.sort_index().copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("Seria STL trebuie să aibă DatetimeIndex.")
    s = s.asfreq("MS")
    s = s.interpolate(method="time").ffill().bfill()
    if len(s.dropna()) < 24:
        raise ValueError("Sunt necesare cel puțin 24 observații lunare pentru STL.")
    return STL(s, period=period, robust=True).fit()


def stationarity_metrics_from_series(s):

    s = pd.Series(s).dropna()

    if len(s) < 10:
        return {
            "p_value": None,
            "stationary": "Necunoscut"
        }

    if s.nunique() <= 1:
        return {
            "p_value": None,
            "stationary": "Constantă"
        }

    try:

        result = adfuller(s, autolag="AIC")

        p_value = result[1]

        stationary = (
            "Staționară"
            if p_value < 0.05
            else "Nestaționară"
        )

        return {
            "p_value": p_value,
            "stationary": stationary
        }

    except Exception:

        return {
            "p_value": None,
            "stationary": "Eroare"
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

    anomalies = count_anomalies_in_series(
        series,
        period=12
    )

    x = np.arange(len(series))

    slope = linregress(
        x,
        series.values
    ).slope

    return {
        "mean": float(series.mean()),

        "std": float(series.std(ddof=0)),

        "min": float(series.min()),

        "max": float(series.max()),

        "amplitude": float(
            series.max() - series.min()
        ),

        "trend_slope": float(slope),

        "anomaly_count": int(anomalies),
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

def amss_distance(series_a, series_b):
    """
    Calculează o distanță AMSS simplificată între două serii temporale.

    Ideea este să comparăm forma generală a seriilor după normalizare.
    Valorile mici indică profile temporale asemănătoare, iar valorile mari
    indică diferențe mai pronunțate între formele seriilor.
    """
    import numpy as np
    import pandas as pd

    a = pd.Series(series_a).astype(float)
    b = pd.Series(series_b).astype(float)

    min_len = min(len(a), len(b))

    if min_len == 0:
        return float("nan")

    a = a.iloc[:min_len].replace([np.inf, -np.inf], np.nan)
    b = b.iloc[:min_len].replace([np.inf, -np.inf], np.nan)

    a = a.interpolate().bfill().ffill()
    b = b.interpolate().bfill().ffill()

    if a.isna().any() or b.isna().any():
        return float("nan")

    a_std = a.std()
    b_std = b.std()

    if a_std == 0 or b_std == 0:
        return float(abs(a.mean() - b.mean()))

    a_norm = (a - a.mean()) / a_std
    b_norm = (b - b.mean()) / b_std

    shape_distance = np.mean(np.abs(a_norm - b_norm))

    slope_a = np.diff(a_norm)
    slope_b = np.diff(b_norm)

    if len(slope_a) == 0 or len(slope_b) == 0:
        slope_distance = 0.0
    else:
        slope_distance = np.mean(np.abs(slope_a - slope_b))

    return float(0.7 * shape_distance + 0.3 * slope_distance)


def pairwise_amss_matrix(profile_rows):
    """
    Construiește matricea AMSS între profile temporale.

    profile_rows trebuie să fie o listă de forma:
    [
        ("Cluster 1", series_1),
        ("Cluster 2", series_2),
        ...
    ]
    """
    import numpy as np

    n = len(profile_rows)

    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = amss_distance(
                    profile_rows[i][1],
                    profile_rows[j][1],
                )

    return matrix