from flask import Blueprint, render_template, request
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

from services.ndvi_service import load_ndvi, pretty_site_name
from utils.nav import render_nav
from utils.page import figure_card
from utils.ts_utils import prepare_monthly_series, mae, rmse, mape

forecast_bp = Blueprint("forecast", __name__)


def choose_forecast_config(site_code: str) -> dict:
    if site_code == "UrbanCentral":
        return {
            "model_name": "SARIMA(1,0,1)(1,0,0,12)",
            "order": (1, 0, 1),
            "seasonal_order": (1, 0, 0, 12),
            "seasonal": True,
        }
    if site_code == "ParcBucuresti":
        return {
            "model_name": "SARIMA(1,1,1)(1,1,1,12)",
            "order": (1, 1, 1),
            "seasonal_order": (1, 1, 1, 12),
            "seasonal": True,
        }
    return {
        "model_name": "SARIMA(1,1,1)(1,1,1,12)",
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 12),
        "seasonal": True,
    }


def clip_ndvi_forecast(series: pd.Series, lower: float = -0.05, upper: float = 1.0) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def seasonal_baseline_forecast(series: pd.Series, steps: int = 12, season_length: int = 12) -> pd.Series:
    last_season = series.iloc[-season_length:].copy()
    future_index = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1),
        periods=steps,
        freq="MS"
    )

    values = []
    for i in range(steps):
        values.append(float(last_season.iloc[i % season_length]))

    return pd.Series(values, index=future_index)


def arima_forecast_for_site(site_code: str, sub: pd.DataFrame, test_size: int = 12, future_steps: int = 12):
    series = prepare_monthly_series(sub)
    if len(series) < 36:
        raise ValueError("Seria este prea scurtă pentru o evaluare forecast robustă.")
    if len(series) <= test_size + 12:
        raise ValueError("Nu există suficiente date pentru split train/test.")

    cfg = choose_forecast_config(site_code)
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    if cfg["seasonal"]:
        model = SARIMAX(
            train,
            order=cfg["order"],
            seasonal_order=cfg["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    else:
        model = SARIMAX(
            train,
            order=cfg["order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

    fit = model.fit(disp=False)

    pred_res = fit.get_forecast(steps=len(test))
    pred_test = pred_res.predicted_mean
    pred_test.index = test.index
    pred_test = clip_ndvi_forecast(pred_test)

    pred_conf = pred_res.conf_int()
    pred_conf.index = test.index

    if cfg["seasonal"]:
        full_model = SARIMAX(
            series,
            order=cfg["order"],
            seasonal_order=cfg["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
    else:
        full_model = SARIMAX(
            series,
            order=cfg["order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

    full_fit = full_model.fit(disp=False)
    future_res = full_fit.get_forecast(steps=future_steps)
    future_forecast = future_res.predicted_mean
    future_forecast = clip_ndvi_forecast(future_forecast)

    future_conf = future_res.conf_int()

    return {
        "series": series,
        "train": train,
        "test": test,
        "pred_test": pred_test,
        "pred_conf": pred_conf,
        "future_forecast": future_forecast,
        "future_conf": future_conf,
        "mae": mae(test, pred_test),
        "rmse": rmse(test, pred_test),
        "mape": mape(test, pred_test),
        "model_name": cfg["model_name"],
    }


def lstm_forecast_for_site(site_code: str, sub: pd.DataFrame, test_size: int = 12, window: int = 12, future_steps: int = 12):
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception as exc:
        raise RuntimeError(
            "Pentru /forecast-lstm trebuie instalate pachetele tensorflow și scikit-learn."
        ) from exc

    series = prepare_monthly_series(sub)
    if len(series) < 48:
        raise ValueError("Seria este prea scurtă pentru un exemplu LSTM stabil.")

    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    train_scaled = scaled[:-test_size]
    test_scaled = scaled[-test_size:]

    if len(train_scaled) <= window:
        raise ValueError("Date insuficiente pentru fereastra LSTM.")

    def make_sequences(arr, seq_len):
        X, y = [], []
        for i in range(seq_len, len(arr)):
            X.append(arr[i - seq_len:i])
            y.append(arr[i])
        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(train_scaled, window)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(32, input_shape=(window, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="loss",
        patience=12,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        epochs=120,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop],
    )

    # test prediction
    history_seq = list(train_scaled[-window:])
    pred_test_scaled = []

    for actual in test_scaled:
        x_input = np.array(history_seq[-window:]).reshape((1, window, 1))
        pred = model.predict(x_input, verbose=0)[0, 0]
        pred_test_scaled.append(pred)
        history_seq.append(actual)

    pred_test = scaler.inverse_transform(np.array(pred_test_scaled).reshape(-1, 1)).flatten()
    pred_test = pd.Series(pred_test, index=series.index[-test_size:])
    pred_test = clip_ndvi_forecast(pred_test)

    # full model
    X_full, y_full = make_sequences(scaled, window)
    X_full = X_full.reshape((X_full.shape[0], X_full.shape[1], 1))

    full_model = Sequential([
        LSTM(32, input_shape=(window, 1)),
        Dense(1),
    ])
    full_model.compile(optimizer="adam", loss="mse")
    full_model.fit(
        X_full,
        y_full,
        epochs=120,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop],
    )

    # recursive future
    future_seq = list(scaled[-window:])
    future_scaled = []

    for _ in range(future_steps):
        x_input = np.array(future_seq[-window:]).reshape((1, window, 1))
        pred = full_model.predict(x_input, verbose=0)[0, 0]
        future_scaled.append(pred)
        future_seq.append(pred)

    future_vals_model = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()

    future_index = pd.date_range(
        start=series.index[-1] + pd.offsets.MonthBegin(1),
        periods=future_steps,
        freq="MS"
    )

    future_model_series = pd.Series(future_vals_model, index=future_index)

    # seasonal baseline blend to avoid dead straight collapse
    seasonal_baseline = seasonal_baseline_forecast(series, steps=future_steps, season_length=12)
    future_forecast = 0.55 * seasonal_baseline + 0.45 * future_model_series
    future_forecast = clip_ndvi_forecast(future_forecast)

    test = series.iloc[-test_size:]

    return {
        "series": series,
        "train": series.iloc[:-test_size],
        "test": test,
        "pred_test": pred_test,
        "future_forecast": future_forecast,
        "mae": mae(test, pred_test),
        "rmse": rmse(test, pred_test),
        "mape": mape(test, pred_test),
        "model_name": f"LSTM(window={window}, units=32)",
    }


@forecast_bp.route("/forecast-arima")
def forecast_arima_page():
    df = load_ndvi()
    summary_rows = []
    sections = []

    for idx, (site, sub) in enumerate(df.groupby("site"), start=1):
        try:
            result = arima_forecast_for_site(site_code=site, sub=sub, test_size=12, future_steps=12)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=result["train"].index,
                y=result["train"].values,
                mode="lines+markers",
                name="Train"
            ))

            fig.add_trace(go.Scatter(
                x=result["test"].index,
                y=result["test"].values,
                mode="lines+markers",
                name="Test"
            ))

            fig.add_trace(go.Scatter(
                x=result["pred_test"].index,
                y=result["pred_test"].values,
                mode="lines+markers",
                name="Predicție pe test"
            ))

            future_conf = result["future_conf"]
            lower = future_conf.iloc[:, 0].clip(lower=-0.05, upper=1.0)
            upper = future_conf.iloc[:, 1].clip(lower=-0.05, upper=1.0)

            fig.add_trace(go.Scatter(
                x=list(lower.index) + list(upper.index[::-1]),
                y=list(lower.values) + list(upper.values[::-1]),
                fill="toself",
                fillcolor="rgba(120,120,180,0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Interval de încredere"
            ))

            fig.add_trace(go.Scatter(
                x=result["future_forecast"].index,
                y=result["future_forecast"].values,
                mode="lines+markers",
                line=dict(dash="dash"),
                name="Forecast viitor (12 luni)"
            ))

            sections.append(
                figure_card(
                    fig,
                    f"Forecast ARIMA/SARIMA – {pretty_site_name(site)}",
                    (
                        f"Model: {result['model_name']} | "
                        f"MAE: {result['mae']:.4f} | "
                        f"RMSE: {result['rmse']:.4f} | "
                        f"MAPE: {result['mape']:.2f}%"
                    ),
                    section_id=f"forecast_arima_{idx}",
                    yaxis_title="NDVI [0–1]",
                )
            )

            summary_rows.append({
                "site": pretty_site_name(site),
                "model": result["model_name"],
                "mae": round(result["mae"], 4),
                "rmse": round(result["rmse"], 4),
                "mape": round(result["mape"], 2),
                "status": "OK",
            })

        except Exception as exc:
            summary_rows.append({
                "site": pretty_site_name(site),
                "model": "n/a",
                "mae": "n/a",
                "rmse": "n/a",
                "mape": "n/a",
                "status": f"Eroare: {exc}",
            })

    table_rows = ""
    for row in summary_rows:
        table_rows += f"""
        <tr>
          <td>{row["site"]}</td>
          <td>{row["model"]}</td>
          <td>{row["mae"]}</td>
          <td>{row["rmse"]}</td>
          <td>{row["mape"]}</td>
          <td>{row["status"]}</td>
        </tr>
        """

    return render_template(
        "base.html",
        title="Forecast ARIMA",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
          <h1>Forecast ARIMA / SARIMA</h1>
          <p class="muted">
            Pentru modelele ARIMA/SARIMA este afișat și intervalul de încredere al forecast-ului,
            astfel încât proiecția viitoare să nu fie interpretată ca o valoare exactă.
          </p>
          <div class="table-wrap">
            <table class="stats-table">
              <thead>
                <tr>
                  <th>Sit</th>
                  <th>Model</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>MAPE (%)</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>{table_rows}</tbody>
            </table>
          </div>
        </section>
        {''.join(sections)}
        """,
    )


@forecast_bp.route("/forecast-lstm")
def forecast_lstm_page():
    df = load_ndvi()
    summary_rows = []
    sections = []

    for idx, (site, sub) in enumerate(df.groupby("site"), start=1):
        try:
            result = lstm_forecast_for_site(site_code=site, sub=sub, test_size=12, window=12, future_steps=12)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result["train"].index,
                y=result["train"].values,
                mode="lines+markers",
                name="Train"
            ))
            fig.add_trace(go.Scatter(
                x=result["test"].index,
                y=result["test"].values,
                mode="lines+markers",
                name="Test"
            ))
            fig.add_trace(go.Scatter(
                x=result["pred_test"].index,
                y=result["pred_test"].values,
                mode="lines+markers",
                name="Predicție pe test"
            ))
            fig.add_trace(go.Scatter(
                x=result["future_forecast"].index,
                y=result["future_forecast"].values,
                mode="lines+markers",
                line=dict(dash="dash"),
                name="Forecast viitor (12 luni)"
            ))

            sections.append(
                figure_card(
                    fig,
                    f"Forecast LSTM – {pretty_site_name(site)}",
                    (
                        f"Model: {result['model_name']} | "
                        f"MAE: {result['mae']:.4f} | "
                        f"RMSE: {result['rmse']:.4f} | "
                        f"MAPE: {result['mape']:.2f}%"
                    ),
                    section_id=f"forecast_lstm_{idx}",
                    yaxis_title="NDVI [0–1]",
                )
            )

            summary_rows.append({
                "site": pretty_site_name(site),
                "model": result["model_name"],
                "mae": round(result["mae"], 4),
                "rmse": round(result["rmse"], 4),
                "mape": round(result["mape"], 2),
                "status": "OK",
            })

        except Exception as exc:
            summary_rows.append({
                "site": pretty_site_name(site),
                "model": "n/a",
                "mae": "n/a",
                "rmse": "n/a",
                "mape": "n/a",
                "status": f"Eroare: {exc}",
            })

    table_rows = ""
    for row in summary_rows:
        table_rows += f"""
        <tr>
          <td>{row["site"]}</td>
          <td>{row["model"]}</td>
          <td>{row["mae"]}</td>
          <td>{row["rmse"]}</td>
          <td>{row["mape"]}</td>
          <td>{row["status"]}</td>
        </tr>
        """

    return render_template(
        "base.html",
        title="Forecast LSTM",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
          <h1>Forecast LSTM</h1>
          <p class="muted">
            Forecast-ul viitor este obținut prin LSTM și stabilizat printr-o componentă sezonieră recentă,
            pentru a evita proiecțiile artificiale complet monotone.
          </p>
          <div class="table-wrap">
            <table class="stats-table">
              <thead>
                <tr>
                  <th>Sit</th>
                  <th>Model</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>MAPE (%)</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>{table_rows}</tbody>
            </table>
          </div>
        </section>
        {''.join(sections)}
        """,
    )