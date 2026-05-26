from flask import Blueprint, render_template, request
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

from services.indices_service import (
    list_indices,
    load_index_dataframe,
)

from utils.nav import render_nav
from utils.page import figure_card
from utils.ts_utils import (
    mae,
    rmse,
    mape,
)

forecast_bp = Blueprint("forecast", __name__)

ROI_VALUES = ["roi1", "roi2"]
DEFAULT_INDICES = ["NDVI", "NDMI", "SAVI", "AVI", "EVI", "GNDVI"]

INDEX_LIMITS = {
    "NDVI": (0.0, 1.0),
    "SAVI": (0.0, 1.0),
    "GNDVI": (0.0, 1.0),
    "EVI": (0.0, 1.0),
    "NDMI": (-1.0, 1.0),
    "AVI": (0.0, None),
}

DISPLAY_HISTORY_MONTHS = 120
FIGURE_HEIGHT = 680

DEFAULT_FORECAST_STEPS = 12
MAX_FORECAST_STEPS = 72

ANOMALY_Z_THRESHOLD = 3.0
ANOMALY_ROLLING_WINDOW = 12

SEASON_LENGTH = 12
LSTM_WINDOW = 12


def clean_series(series: pd.Series) -> pd.Series:
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    series = series[~series.index.duplicated(keep="first")]
    series = series.sort_index()
    series = series.asfreq("MS")
    series = series.interpolate(method="time").bfill().ffill()
    return series.astype(float)


def clip_forecast(series: pd.Series, index_name: str) -> pd.Series:
    lower, upper = INDEX_LIMITS.get(index_name.upper(), (None, None))
    return series.clip(lower=lower, upper=upper)


def get_available_indices() -> list[str]:
    available_indices = list_indices()

    if not available_indices:
        available_indices = DEFAULT_INDICES

    return [idx.upper() for idx in available_indices]


def get_selected_index(available_indices: list[str]) -> str:
    selected_index = request.args.get("index", "NDVI").upper()

    if selected_index not in available_indices:
        selected_index = available_indices[0] if available_indices else "NDVI"

    return selected_index


def get_selected_roi() -> str:
    selected_roi = request.args.get("roi", "roi1").lower()

    if selected_roi not in ROI_VALUES:
        selected_roi = "roi1"

    return selected_roi


def prepare_roi_series(index_name: str, roi: str) -> pd.Series:
    df = load_index_dataframe(index_name)

    required_columns = {"date", "roi", "value"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Fișierul pentru {index_name} nu conține coloanele necesare: "
            f"{', '.join(sorted(missing_columns))}."
        )

    df = df[df["roi"].str.lower() == roi.lower()].copy()

    if df.empty:
        raise ValueError(f"Nu există date pentru {index_name} - {roi}.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])

    if df.empty:
        raise ValueError(f"Datele pentru {index_name} - {roi} nu sunt valide.")

    series = (
        df.groupby("date")["value"]
        .mean()
        .sort_index()
    )

    return clean_series(series)


def get_target_date(series: pd.Series) -> tuple[str, pd.Timestamp, int, bool]:
    raw_target_date = request.args.get("target_date")
    last_date = series.index[-1]
    default_target = last_date + pd.DateOffset(months=DEFAULT_FORECAST_STEPS)

    try:
        if raw_target_date:
            target_dt = pd.to_datetime(raw_target_date + "-01")
        else:
            target_dt = default_target
    except Exception:
        target_dt = default_target

    months_ahead = (
        (target_dt.year - last_date.year) * 12
        + (target_dt.month - last_date.month)
    )

    if months_ahead < 1:
        target_dt = default_target
        months_ahead = DEFAULT_FORECAST_STEPS

    horizon_was_limited = False

    if months_ahead > MAX_FORECAST_STEPS:
        target_dt = last_date + pd.DateOffset(months=MAX_FORECAST_STEPS)
        months_ahead = MAX_FORECAST_STEPS
        horizon_was_limited = True

    return target_dt.strftime("%Y-%m"), target_dt, months_ahead, horizon_was_limited


def get_future_steps(months_ahead: int) -> int:
    return min(
        max(months_ahead, DEFAULT_FORECAST_STEPS),
        MAX_FORECAST_STEPS,
    )


def get_test_size(series: pd.Series) -> int:
    if len(series) >= 60:
        return 12

    if len(series) >= 48:
        return 9

    if len(series) >= 36:
        return 6

    return max(3, len(series) // 5)


def split_train_test(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    test_size = get_test_size(series)
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    return train, test


def calculate_metrics(test: pd.Series, pred_test: pd.Series) -> dict:
    return {
        "mae": mae(test, pred_test),
        "rmse": rmse(test, pred_test),
        "mape": mape(test, pred_test),
    }


def detect_anomalies_rolling_zscore(
    series: pd.Series,
    window: int = ANOMALY_ROLLING_WINDOW,
    threshold: float = ANOMALY_Z_THRESHOLD,
) -> pd.Series:
    rolling_median = series.rolling(
        window=window,
        center=True,
        min_periods=max(3, window // 2),
    ).median()

    residual = series - rolling_median

    rolling_mad = residual.abs().rolling(
        window=window,
        center=True,
        min_periods=max(3, window // 2),
    ).median()

    robust_sigma = 1.4826 * rolling_mad
    robust_sigma = robust_sigma.replace(0, np.nan).bfill().ffill()

    z_score = residual.abs() / robust_sigma
    anomaly_mask = z_score > threshold

    return anomaly_mask.fillna(False).astype(bool)


def preprocess_anomalies(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    anomaly_mask = detect_anomalies_rolling_zscore(series)

    cleaned = series.copy()
    cleaned.loc[anomaly_mask] = np.nan
    cleaned = cleaned.interpolate(method="time").bfill().ffill()

    return cleaned, anomaly_mask


def choose_sarima_config(series: pd.Series) -> dict:
    if len(series) >= 36:
        return {
            "model_name": "SARIMA(1,0,1)(1,1,1,12)",
            "order": (1, 0, 1),
            "seasonal_order": (1, 1, 1, 12),
        }

    return {
        "model_name": "ARIMA(1,0,1)",
        "order": (1, 0, 1),
        "seasonal_order": (0, 0, 0, 0),
    }


def empirical_confidence_interval(
    forecast: pd.Series,
    residuals: pd.Series,
    index_name: str,
) -> pd.DataFrame:
    residual_std = float(residuals.std(ddof=0))

    if not np.isfinite(residual_std) or residual_std == 0:
        residual_std = max(float(residuals.abs().mean()), 1e-6)

    margin = 1.96 * residual_std

    lower = pd.Series(forecast.values - margin, index=forecast.index)
    upper = pd.Series(forecast.values + margin, index=forecast.index)

    lower = clip_forecast(lower, index_name)
    upper = clip_forecast(upper, index_name)

    return pd.DataFrame({
        "lower": lower,
        "upper": upper,
    })


def sarima_forecast_on_series(
    modeling_series: pd.Series,
    evaluation_series: pd.Series,
    index_name: str,
    future_steps: int,
    model_name_suffix: str,
) -> dict:
    if len(modeling_series) < 24:
        raise ValueError(
            "Seria este prea scurtă pentru ARIMA/SARIMA. "
            "Sunt necesare cel puțin 24 de observații lunare."
        )

    cfg = choose_sarima_config(modeling_series)

    train_model, test_model = split_train_test(modeling_series)
    _, test_real = split_train_test(evaluation_series)

    model = SARIMAX(
        train_model,
        order=cfg["order"],
        seasonal_order=cfg["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fit = model.fit(disp=False)

    pred_test = fit.get_forecast(steps=len(test_model)).predicted_mean
    pred_test.index = test_model.index
    pred_test = clip_forecast(pred_test, index_name)

    full_model = SARIMAX(
        modeling_series,
        order=cfg["order"],
        seasonal_order=cfg["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    full_fit = full_model.fit(disp=False)
    future_res = full_fit.get_forecast(steps=future_steps)

    future_forecast = future_res.predicted_mean
    future_forecast = clip_forecast(future_forecast, index_name)

    future_conf_raw = future_res.conf_int()
    future_conf_raw.index = future_forecast.index

    future_conf = pd.DataFrame({
        "lower": clip_forecast(future_conf_raw.iloc[:, 0], index_name),
        "upper": clip_forecast(future_conf_raw.iloc[:, 1], index_name),
    })

    metrics = calculate_metrics(test_real, pred_test)

    return {
        "series": evaluation_series,
        "modeling_series": modeling_series,
        "train": train_model,
        "test": test_real,
        "pred_test": pred_test,
        "future_forecast": future_forecast,
        "future_conf": future_conf,
        "model_name": f"{cfg['model_name']} – {model_name_suffix}",
        "model_type": "SARIMA",
        "interval_type": "interval statistic",
        "warning": "",
        **metrics,
    }


def import_tensorflow_or_raise():
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        return tf, Sequential, LSTM, Dense, Dropout, EarlyStopping
    except Exception:
        raise RuntimeError(
            "Modelul LSTM nu a putut fi antrenat în mediul curent. "
            "Pentru acest modul sunt necesare dependențele specifice modelării deep learning."
        )


def build_lstm_training_data(
    series: pd.Series,
    value_scaler: MinMaxScaler,
    residual_scaler: MinMaxScaler,
    window: int = LSTM_WINDOW,
    season_length: int = SEASON_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    values = series.values.reshape(-1, 1)
    scaled_values = value_scaler.transform(values)

    X, y = [], []
    start_idx = max(window, season_length)

    for i in range(start_idx, len(series)):
        X.append(scaled_values[i - window:i, 0])

        seasonal_reference = float(series.iloc[i - season_length])
        residual = float(series.iloc[i] - seasonal_reference)
        y.append(residual)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    if len(X) == 0:
        return X, y

    y_scaled = residual_scaler.transform(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y_scaled


def fit_residual_scaler(series: pd.Series, season_length: int = SEASON_LENGTH) -> MinMaxScaler:
    residuals = []

    for i in range(season_length, len(series)):
        residuals.append(float(series.iloc[i] - series.iloc[i - season_length]))

    if not residuals:
        residuals = [0.0]

    residuals = np.array(residuals).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(residuals)

    return scaler


def train_lstm_model(
    train_series: pd.Series,
    value_scaler: MinMaxScaler,
    residual_scaler: MinMaxScaler,
    window: int = LSTM_WINDOW,
    season_length: int = SEASON_LENGTH,
):
    tf, Sequential, LSTM, Dense, Dropout, EarlyStopping = import_tensorflow_or_raise()

    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, y_train = build_lstm_training_data(
        series=train_series,
        value_scaler=value_scaler,
        residual_scaler=residual_scaler,
        window=window,
        season_length=season_length,
    )

    if len(X_train) < 10:
        raise ValueError(
            "Seria are prea puține ferestre de antrenare pentru LSTM. "
            f"Ferestre disponibile: {len(X_train)}. Sunt necesare cel puțin 10."
        )

    model = Sequential([
        LSTM(32, input_shape=(window, 1), return_sequences=False),
        Dropout(0.10),
        Dense(12, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    early_stop = EarlyStopping(
        monitor="loss",
        patience=12,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop],
        shuffle=False,
    )

    return model


def predict_lstm_residual(
    model,
    history_values: list[float],
    value_scaler: MinMaxScaler,
    residual_scaler: MinMaxScaler,
    residual_limit: float,
    window: int = LSTM_WINDOW,
) -> float:
    history_array = np.array(history_values).reshape(-1, 1)
    scaled_history = value_scaler.transform(history_array)

    current_window = scaled_history[-window:, 0].reshape(1, window, 1)
    residual_scaled = float(model.predict(current_window, verbose=0)[0][0])
    residual = residual_scaler.inverse_transform(np.array([[residual_scaled]]))[0][0]

    residual = float(np.clip(residual, -residual_limit, residual_limit))

    return residual


def get_residual_limit(series: pd.Series, season_length: int = SEASON_LENGTH) -> float:
    residuals = []

    for i in range(season_length, len(series)):
        residuals.append(float(series.iloc[i] - series.iloc[i - season_length]))

    if not residuals:
        return 0.05

    residuals = np.array(residuals)
    limit = np.nanpercentile(np.abs(residuals), 75)

    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanstd(series.values)) * 0.5

    if not np.isfinite(limit) or limit <= 0:
        limit = 0.05

    return float(limit)


def lstm_test_predict(
    model,
    modeling_series: pd.Series,
    train_size: int,
    test_size: int,
    value_scaler: MinMaxScaler,
    residual_scaler: MinMaxScaler,
    residual_limit: float,
    index_name: str,
    window: int = LSTM_WINDOW,
    season_length: int = SEASON_LENGTH,
) -> pd.Series:
    predictions = []
    prediction_index = modeling_series.index[train_size:train_size + test_size]

    for i in range(test_size):
        current_pos = train_size + i
        history = modeling_series.iloc[:current_pos].values.astype(float).tolist()

        residual = predict_lstm_residual(
            model=model,
            history_values=history,
            value_scaler=value_scaler,
            residual_scaler=residual_scaler,
            residual_limit=residual_limit,
            window=window,
        )

        seasonal_reference = float(modeling_series.iloc[current_pos - season_length])
        predicted_value = seasonal_reference + residual
        predictions.append(predicted_value)

    pred = pd.Series(predictions, index=prediction_index)
    return clip_forecast(pred, index_name)


def lstm_future_forecast(
    model,
    modeling_series: pd.Series,
    future_steps: int,
    value_scaler: MinMaxScaler,
    residual_scaler: MinMaxScaler,
    residual_limit: float,
    index_name: str,
    window: int = LSTM_WINDOW,
    season_length: int = SEASON_LENGTH,
) -> pd.Series:
    history = modeling_series.values.astype(float).tolist()

    future_index = pd.date_range(
        start=modeling_series.index[-1] + pd.offsets.MonthBegin(1),
        periods=future_steps,
        freq="MS",
    )

    predictions = []

    for _ in range(future_steps):
        residual = predict_lstm_residual(
            model=model,
            history_values=history,
            value_scaler=value_scaler,
            residual_scaler=residual_scaler,
            residual_limit=residual_limit,
            window=window,
        )

        seasonal_reference = float(history[-season_length])
        predicted_value = seasonal_reference + residual

        clipped = clip_forecast(pd.Series([predicted_value]), index_name)
        predicted_value = float(clipped.iloc[0])

        predictions.append(predicted_value)
        history.append(predicted_value)

    return pd.Series(predictions, index=future_index)


def seasonal_reference_forecast(
    series: pd.Series,
    steps: int,
    index_name: str,
    season_length: int = SEASON_LENGTH,
) -> pd.Series:
    if len(series) >= season_length:
        seasonal_values = series.iloc[-season_length:].values.astype(float)
    else:
        seasonal_values = series.values.astype(float)

    future_index = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1),
        periods=steps,
        freq="MS",
    )

    values = []

    for i in range(steps):
        values.append(float(seasonal_values[i % len(seasonal_values)]))

    forecast = pd.Series(values, index=future_index)
    return clip_forecast(forecast, index_name)


def seasonal_reference_test(
    modeling_series: pd.Series,
    train_size: int,
    test_size: int,
    index_name: str,
    season_length: int = SEASON_LENGTH,
) -> pd.Series:
    prediction_index = modeling_series.index[train_size:train_size + test_size]
    values = []

    for i in range(test_size):
        current_pos = train_size + i

        if current_pos - season_length >= 0:
            values.append(float(modeling_series.iloc[current_pos - season_length]))
        else:
            values.append(float(modeling_series.iloc[current_pos - 1]))

    pred = pd.Series(values, index=prediction_index)
    return clip_forecast(pred, index_name)


def blend_lstm_with_seasonality(
    lstm_series: pd.Series,
    seasonal_series: pd.Series,
    index_name: str,
    seasonal_weight: float = 0.70,
) -> pd.Series:
    common_index = lstm_series.index.intersection(seasonal_series.index)

    blended = (
        seasonal_weight * seasonal_series.loc[common_index]
        + (1.0 - seasonal_weight) * lstm_series.loc[common_index]
    )

    blended = pd.Series(blended.values, index=common_index)
    return clip_forecast(blended, index_name)


def lstm_forecast_on_series(
    modeling_series: pd.Series,
    evaluation_series: pd.Series,
    index_name: str,
    future_steps: int,
    model_name_suffix: str,
) -> dict:
    if len(modeling_series) < 36:
        raise ValueError(
            "Seria este prea scurtă pentru LSTM. "
            "Sunt necesare cel puțin 36 de observații lunare."
        )

    train_model, test_model = split_train_test(modeling_series)
    _, test_real = split_train_test(evaluation_series)

    if len(train_model) < LSTM_WINDOW + SEASON_LENGTH + 10:
        raise ValueError(
            "Seria are prea puține observații pentru LSTM sezonier. "
            "Sunt necesare mai multe observații lunare pentru antrenare stabilă."
        )

    value_scaler = MinMaxScaler()
    value_scaler.fit(train_model.values.reshape(-1, 1))

    residual_scaler = fit_residual_scaler(train_model)
    residual_limit = get_residual_limit(train_model)

    model = train_lstm_model(
        train_series=train_model,
        value_scaler=value_scaler,
        residual_scaler=residual_scaler,
    )

    lstm_pred_test = lstm_test_predict(
        model=model,
        modeling_series=modeling_series,
        train_size=len(train_model),
        test_size=len(test_model),
        value_scaler=value_scaler,
        residual_scaler=residual_scaler,
        residual_limit=residual_limit,
        index_name=index_name,
    )

    seasonal_pred_test = seasonal_reference_test(
        modeling_series=modeling_series,
        train_size=len(train_model),
        test_size=len(test_model),
        index_name=index_name,
    )

    pred_test = blend_lstm_with_seasonality(
        lstm_series=lstm_pred_test,
        seasonal_series=seasonal_pred_test,
        index_name=index_name,
        seasonal_weight=0.65,
    )

    lstm_future = lstm_future_forecast(
        model=model,
        modeling_series=modeling_series,
        future_steps=future_steps,
        value_scaler=value_scaler,
        residual_scaler=residual_scaler,
        residual_limit=residual_limit,
        index_name=index_name,
    )

    seasonal_future = seasonal_reference_forecast(
        series=modeling_series,
        steps=future_steps,
        index_name=index_name,
    )

    future_forecast = blend_lstm_with_seasonality(
        lstm_series=lstm_future,
        seasonal_series=seasonal_future,
        index_name=index_name,
        seasonal_weight=0.75,
    )

    metrics = calculate_metrics(test_real, pred_test)

    return {
        "series": evaluation_series,
        "modeling_series": modeling_series,
        "train": train_model,
        "test": test_real,
        "pred_test": pred_test,
        "future_forecast": future_forecast,
        "future_conf": None,
        "model_name": f"LSTM hibrid sezonier, window={LSTM_WINDOW} – {model_name_suffix}",
        "model_type": "LSTM",
        "interval_type": "evaluare prin metrici",
        "warning": "",
        **metrics,
    }

def build_comparison(
    raw_series: pd.Series,
    cleaned_series: pd.Series,
    index_name: str,
    future_steps: int,
    forecast_type: str,
) -> dict:
    if forecast_type == "arima":
        raw_result = sarima_forecast_on_series(
            modeling_series=raw_series,
            evaluation_series=raw_series,
            index_name=index_name,
            future_steps=future_steps,
            model_name_suffix="serie brută",
        )

        cleaned_result = sarima_forecast_on_series(
            modeling_series=cleaned_series,
            evaluation_series=raw_series,
            index_name=index_name,
            future_steps=future_steps,
            model_name_suffix="serie preprocesată",
        )

        return {
            "raw": raw_result,
            "cleaned": cleaned_result,
            "model_name": "SARIMA comparativ",
            "model_type": "SARIMA",
            "warning": "",
        }

    raw_result = lstm_forecast_on_series(
        modeling_series=raw_series,
        evaluation_series=raw_series,
        index_name=index_name,
        future_steps=future_steps,
        model_name_suffix="serie brută",
    )

    cleaned_result = lstm_forecast_on_series(
        modeling_series=cleaned_series,
        evaluation_series=raw_series,
        index_name=index_name,
        future_steps=future_steps,
        model_name_suffix="serie preprocesată",
    )

    return {
        "raw": raw_result,
        "cleaned": cleaned_result,
        "model_name": "LSTM comparativ",
        "model_type": "LSTM",
        "warning": "",
    }


def get_display_scale(series: pd.Series, index_name: str) -> tuple[float, str]:
    max_abs = float(np.nanmax(np.abs(series.values)))

    if index_name.upper() == "AVI" and max_abs < 0.01:
        return 1000.0, f"{index_name} × 1000"

    if max_abs < 0.01:
        return 1000.0, f"{index_name} × 1000"

    return 1.0, index_name


def scale_series(series: pd.Series, scale: float) -> pd.Series:
    return series * scale


def get_forecast_value(future: pd.Series, target_dt: pd.Timestamp) -> float:
    if target_dt in future.index:
        return float(future.loc[target_dt])

    nearest_idx = future.index.get_indexer([target_dt], method="nearest")[0]
    return float(future.iloc[nearest_idx])


def interpret_forecast(
    index_name: str,
    value_raw: float,
    value_cleaned: float,
    target_date: str,
    series: pd.Series,
) -> str:
    historical_mean = float(series.mean())
    historical_min = float(series.min())
    historical_max = float(series.max())

    difference = value_cleaned - value_raw

    if abs(difference) <= max(0.05 * abs(value_raw), 0.000001):
        comparison_text = (
            "Cele două variante produc valori apropiate, ceea ce sugerează că tratarea anomaliilor "
            "nu modifică semnificativ rezultatul final."
        )
    elif difference > 0:
        comparison_text = (
            "Prognoza pe seria preprocesată este mai ridicată decât prognoza pe seria brută, "
            "ceea ce indică influența valorilor extreme asupra modelului inițial."
        )
    else:
        comparison_text = (
            "Prognoza pe seria preprocesată este mai scăzută decât prognoza pe seria brută, "
            "ceea ce indică influența valorilor extreme asupra modelului inițial."
        )

    return f"""
    Pentru luna <strong>{target_date}</strong>, prognoza pe seria brută estimează
    <strong>{index_name} = {value_raw:.6f}</strong>, iar prognoza pe seria preprocesată estimează
    <strong>{index_name} = {value_cleaned:.6f}</strong>.
    <br><br>
    {comparison_text}
    <br><br>
    Media istorică este <strong>{historical_mean:.6f}</strong>, cu valori observate între
    <strong>{historical_min:.6f}</strong> și <strong>{historical_max:.6f}</strong>.
    """


def build_options(values: list[str], selected_value: str) -> str:
    html = ""

    for value in values:
        value_str = str(value)
        selected = "selected" if value_str.upper() == selected_value.upper() else ""
        label = value_str.upper()
        html += f"<option value='{value_str}' {selected}>{label}</option>"

    return html


def add_confidence_interval(
    fig: go.Figure,
    future_conf: pd.DataFrame,
    scale: float,
    name: str,
):
    if future_conf is None or future_conf.empty:
        return

    fig.add_trace(go.Scatter(
        x=future_conf.index,
        y=future_conf["upper"] * scale,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name=f"{name} superior",
    ))

    fig.add_trace(go.Scatter(
        x=future_conf.index,
        y=future_conf["lower"] * scale,
        mode="lines",
        fill="tonexty",
        line=dict(width=0),
        name=f"Interval {name}",
        hoverinfo="skip",
    ))


def add_forecast_start_line(fig: go.Figure, forecast_start: pd.Timestamp):
    forecast_start_str = pd.to_datetime(forecast_start).strftime("%Y-%m-%d")

    fig.add_shape(
        type="line",
        x0=forecast_start_str,
        x1=forecast_start_str,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            dash="dot",
            width=2,
        ),
    )

    fig.add_annotation(
        x=forecast_start_str,
        y=1,
        xref="x",
        yref="paper",
        text="Start forecast",
        showarrow=False,
        yshift=12,
        xanchor="left",
    )


def base_layout(fig: go.Figure, title: str, yaxis_label: str):
    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis_title=yaxis_label,
        hovermode="x unified",
        height=FIGURE_HEIGHT,
        margin=dict(l=60, r=40, t=115, b=75),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
        ),
    )


def build_single_forecast_figure(
    result: dict,
    source_series: pd.Series,
    selected_index: str,
    selected_roi: str,
    target_dt: pd.Timestamp,
    target_date: str,
    target_value: float,
    title: str,
    variant_label: str,
) -> tuple[go.Figure, str]:
    fig = go.Figure()

    display_start = source_series.index[-min(DISPLAY_HISTORY_MONTHS, len(source_series))]

    train_display = result["train"][result["train"].index >= display_start]
    test_display = result["test"][result["test"].index >= display_start]
    pred_display = result["pred_test"][result["pred_test"].index >= display_start]

    combined = pd.concat([
        source_series,
        result["future_forecast"],
    ])

    scale, yaxis_label = get_display_scale(combined, selected_index)

    fig.add_trace(go.Scatter(
        x=train_display.index,
        y=scale_series(train_display, scale),
        mode="lines",
        name=f"Train - {variant_label}",
        line=dict(width=2),
    ))

    fig.add_trace(go.Scatter(
        x=test_display.index,
        y=scale_series(test_display, scale),
        mode="lines+markers",
        name="Test real",
        marker=dict(size=7),
    ))

    fig.add_trace(go.Scatter(
        x=pred_display.index,
        y=scale_series(pred_display, scale),
        mode="lines+markers",
        name="Predicție test",
        marker=dict(size=7),
    ))

    add_confidence_interval(
        fig=fig,
        future_conf=result.get("future_conf"),
        scale=scale,
        name=variant_label,
    )

    fig.add_trace(go.Scatter(
        x=result["future_forecast"].index,
        y=scale_series(result["future_forecast"], scale),
        mode="lines",
        line=dict(dash="dash", width=3),
        name=f"Forecast - {variant_label}",
    ))

    fig.add_trace(go.Scatter(
        x=[target_dt],
        y=[target_value * scale],
        mode="markers+text",
        marker=dict(size=13),
        text=[f"{target_date}: {target_value * scale:.4f}"],
        textposition="top center",
        name="Luna selectată",
    ))

    add_forecast_start_line(fig, result["future_forecast"].index[0])

    base_layout(
        fig=fig,
        title=f"{title} – {selected_index} – {selected_roi.upper()}",
        yaxis_label=yaxis_label,
    )

    fig.update_yaxes(
        rangemode="tozero" if selected_index.upper() == "AVI" else "normal"
    )

    return fig, yaxis_label


def build_comparison_figure(
    comparison: dict,
    raw_series: pd.Series,
    cleaned_series: pd.Series,
    selected_index: str,
    selected_roi: str,
    target_dt: pd.Timestamp,
    target_date: str,
    value_raw: float,
    value_cleaned: float,
    route_title: str,
) -> tuple[go.Figure, str]:
    fig = go.Figure()

    raw_result = comparison["raw"]
    cleaned_result = comparison["cleaned"]

    display_start = raw_series.index[-min(DISPLAY_HISTORY_MONTHS, len(raw_series))]

    raw_display = raw_series[raw_series.index >= display_start]
    cleaned_display = cleaned_series[cleaned_series.index >= display_start]

    combined = pd.concat([
        raw_series,
        cleaned_series,
        raw_result["future_forecast"],
        cleaned_result["future_forecast"],
    ])

    scale, yaxis_label = get_display_scale(combined, selected_index)

    fig.add_trace(go.Scatter(
        x=raw_display.index,
        y=scale_series(raw_display, scale),
        mode="lines",
        name="Istoric brut",
        line=dict(width=2),
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_display.index,
        y=scale_series(cleaned_display, scale),
        mode="lines",
        name="Istoric preprocesat",
        line=dict(width=2, dash="dot"),
    ))

    fig.add_trace(go.Scatter(
        x=raw_result["future_forecast"].index,
        y=scale_series(raw_result["future_forecast"], scale),
        mode="lines",
        line=dict(dash="dash", width=3),
        name="Forecast brut",
    ))

    fig.add_trace(go.Scatter(
        x=cleaned_result["future_forecast"].index,
        y=scale_series(cleaned_result["future_forecast"], scale),
        mode="lines",
        line=dict(dash="dashdot", width=3),
        name="Forecast preprocesat",
    ))

    fig.add_trace(go.Scatter(
        x=[target_dt],
        y=[value_raw * scale],
        mode="markers+text",
        marker=dict(size=12),
        text=[f"Brut: {value_raw * scale:.4f}"],
        textposition="top center",
        name=f"Luna selectată brut {target_date}",
    ))

    fig.add_trace(go.Scatter(
        x=[target_dt],
        y=[value_cleaned * scale],
        mode="markers+text",
        marker=dict(size=12),
        text=[f"Preprocesat: {value_cleaned * scale:.4f}"],
        textposition="bottom center",
        name=f"Luna selectată preprocesat {target_date}",
    ))

    add_forecast_start_line(fig, raw_result["future_forecast"].index[0])

    base_layout(
        fig=fig,
        title=f"{route_title} – comparație finală – {selected_index} – {selected_roi.upper()}",
        yaxis_label=yaxis_label,
    )

    fig.update_yaxes(
        rangemode="tozero" if selected_index.upper() == "AVI" else "normal"
    )

    return fig, yaxis_label


def build_metrics_table(comparison: dict, anomaly_count: int) -> str:
    raw = comparison["raw"]
    cleaned = comparison["cleaned"]

    return f"""
<section class="card reveal active">
    <h2>Comparație metrici</h2>
    <div class="table-wrap">
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Variantă</th>
                    <th>Model</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>MAPE</th>
                    <th>Evaluare</th>
                    <th>Anomalii tratate</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Serie brută</td>
                    <td>{raw["model_name"]}</td>
                    <td>{raw["mae"]:.6f}</td>
                    <td>{raw["rmse"]:.6f}</td>
                    <td>{raw["mape"]:.2f}%</td>
                    <td>{raw["interval_type"]}</td>
                    <td>0</td>
                </tr>
                <tr>
                    <td>Serie preprocesată</td>
                    <td>{cleaned["model_name"]}</td>
                    <td>{cleaned["mae"]:.6f}</td>
                    <td>{cleaned["rmse"]:.6f}</td>
                    <td>{cleaned["mape"]:.2f}%</td>
                    <td>{cleaned["interval_type"]}</td>
                    <td>{anomaly_count}</td>
                </tr>
            </tbody>
        </table>
    </div>
</section>
    """


def build_error_page(route_title: str, route_path: str, error_message: str):
    return render_template(
        "base.html",
        title=route_title,
        nav_html=render_nav(route_path),
        content=f"""
<section class="card reveal active">
    <h1>{route_title}</h1>
    <div class="method-box">
        <strong>Nu s-a putut genera forecast-ul.</strong><br>
        {error_message}
    </div>
</section>
        """,
    )


def build_forecast_page(
    route_title: str,
    route_path: str,
    forecast_type: str,
):
    available_indices = get_available_indices()
    selected_index = get_selected_index(available_indices)
    selected_roi = get_selected_roi()

    try:
        raw_series = prepare_roi_series(selected_index, selected_roi)
        cleaned_series, anomaly_mask = preprocess_anomalies(raw_series)
        anomaly_count = int(anomaly_mask.sum())

        target_date, target_dt, months_ahead, horizon_was_limited = get_target_date(raw_series)
        future_steps = get_future_steps(months_ahead)

        comparison = build_comparison(
            raw_series=raw_series,
            cleaned_series=cleaned_series,
            index_name=selected_index,
            future_steps=future_steps,
            forecast_type=forecast_type,
        )

        raw_future = comparison["raw"]["future_forecast"]
        cleaned_future = comparison["cleaned"]["future_forecast"]

        value_raw = get_forecast_value(raw_future, target_dt)
        value_cleaned = get_forecast_value(cleaned_future, target_dt)

        interpretation = interpret_forecast(
            index_name=selected_index,
            value_raw=value_raw,
            value_cleaned=value_cleaned,
            target_date=target_date,
            series=raw_series,
        )

        raw_fig, raw_yaxis_label = build_single_forecast_figure(
            result=comparison["raw"],
            source_series=raw_series,
            selected_index=selected_index,
            selected_roi=selected_roi,
            target_dt=target_dt,
            target_date=target_date,
            target_value=value_raw,
            title=f"{route_title} – serie brută",
            variant_label="brut",
        )

        cleaned_fig, cleaned_yaxis_label = build_single_forecast_figure(
            result=comparison["cleaned"],
            source_series=cleaned_series,
            selected_index=selected_index,
            selected_roi=selected_roi,
            target_dt=target_dt,
            target_date=target_date,
            target_value=value_cleaned,
            title=f"{route_title} – serie preprocesată",
            variant_label="preprocesat",
        )

        comparison_fig, comparison_yaxis_label = build_comparison_figure(
            comparison=comparison,
            raw_series=raw_series,
            cleaned_series=cleaned_series,
            selected_index=selected_index,
            selected_roi=selected_roi,
            target_dt=target_dt,
            target_date=target_date,
            value_raw=value_raw,
            value_cleaned=value_cleaned,
            route_title=route_title,
        )

    except Exception as exc:
        return build_error_page(
            route_title=route_title,
            route_path=route_path,
            error_message=str(exc),
        )

    index_options = build_options(available_indices, selected_index)
    roi_options = build_options(ROI_VALUES, selected_roi)

    horizon_note = ""

    if horizon_was_limited:
        horizon_note = f"""
        <div class="method-box">
            <strong>Observație:</strong><br>
            Orizontul de prognoză a fost limitat automat la {MAX_FORECAST_STEPS} luni.
            Pentru intervale mari, interpretarea trebuie tratată ca estimare exploratorie.
        </div>
        """

    if forecast_type == "arima":
        interval_note = "Intervalul de incertitudine este generat de modelul statistic."
    else:
        interval_note = "Performanța este evaluată prin MAE, RMSE și MAPE pe perioada de test."

    metrics_table = build_metrics_table(
        comparison=comparison,
        anomaly_count=anomaly_count,
    )

    return render_template(
        "base.html",
        title=route_title,
        nav_html=render_nav(route_path),
        content=f"""
<section class="card reveal active">
    <h1>{route_title}</h1>

    <p class="muted">
        Modulul compară prognoza obținută pe seria brută cu prognoza obținută după tratarea valorilor anomale.
    </p>

    <form method="get" class="method-box">
        <label><strong>Indice spectral:</strong></label><br>
        <select name="index" class="select-input">
            {index_options}
        </select>

        <br><br>

        <label><strong>ROI:</strong></label><br>
        <select name="roi" class="select-input">
            {roi_options}
        </select>

        <br><br>

        <label><strong>Luna țintă:</strong></label><br>
        <input
            type="month"
            name="target_date"
            value="{target_date}"
            class="select-input"
        >

        <br><br>

        <button class="btn btn-primary" type="submit">
            Actualizează forecast
        </button>
    </form>

    <div class="method-box">
        <strong>Model:</strong> {comparison["model_name"]}<br>
        <strong>Tip model:</strong> {comparison["model_type"]}<br>
        <strong>Anomalii detectate și tratate:</strong> {anomaly_count}<br>
        <strong>Metodă detecție:</strong> mediană mobilă + z-score robust<br>
        <strong>Fereastră analizată:</strong> {ANOMALY_ROLLING_WINDOW} luni<br>
        <strong>Prag detecție:</strong> z-score robust &gt; {ANOMALY_Z_THRESHOLD}<br>
        <strong>Observații lunare utilizate:</strong> {len(raw_series)}<br>
        <strong>Perioadă date:</strong> {raw_series.index.min().strftime("%Y-%m")} – {raw_series.index.max().strftime("%Y-%m")}<br>
        <strong>Orizont prognoză:</strong> {future_steps} luni<br>
        <strong>Evaluare:</strong> {interval_note}
    </div>

    {horizon_note}
</section>

<section class="card reveal active">
    <h2>Interpretare predictivă</h2>
    <p class="muted">
        {interpretation}
    </p>
</section>

{figure_card(
    raw_fig,
    f"{route_title} – serie brută – {selected_index} – {selected_roi.upper()}",
    "Modelare directă pe seria brută.",
    section_id=f"{forecast_type}_raw_forecast_fig",
    xaxis_title="Data",
    yaxis_title=raw_yaxis_label,
)}

{figure_card(
    cleaned_fig,
    f"{route_title} – serie preprocesată – {selected_index} – {selected_roi.upper()}",
    "Modelare pe seria preprocesată prin tratarea valorilor anomale.",
    section_id=f"{forecast_type}_cleaned_forecast_fig",
    xaxis_title="Data",
    yaxis_title=cleaned_yaxis_label,
)}

{figure_card(
    comparison_fig,
    f"{route_title} – comparație brut vs preprocesat – {selected_index} – {selected_roi.upper()}",
    "Comparație finală între cele două variante de prognoză.",
    section_id=f"{forecast_type}_comparison_forecast_fig",
    xaxis_title="Data",
    yaxis_title=comparison_yaxis_label,
)}

{metrics_table}

<section class="card reveal active">
    <h2>Notă metodologică</h2>
    <div class="method-box">
        Seria preprocesată nu înlocuiește seria reală, ci permite evaluarea influenței valorilor extreme asupra modelului predictiv.
    </div>
</section>
        """,
    )


@forecast_bp.route("/forecast-arima")
def forecast_arima_page():
    return build_forecast_page(
        route_title="Forecast ARIMA / SARIMA",
        route_path=request.path,
        forecast_type="arima",
    )


@forecast_bp.route("/forecast-lstm")
def forecast_lstm_page():
    return build_forecast_page(
        route_title="Forecast LSTM",
        route_path=request.path,
        forecast_type="lstm",
    )
