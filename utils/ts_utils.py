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