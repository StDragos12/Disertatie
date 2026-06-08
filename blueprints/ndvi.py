from flask import Blueprint, render_template, request
import plotly.graph_objects as go

from services.indices_service import (
    load_index_dataframe,
    list_indices,
    INDEX_DESCRIPTIONS,
    smooth_series,
)
from services.dataset_service import (
    DEMO_DATASET_ID,
    build_dataset_options_html,
    get_dataset_display_name,
    get_dataset_rois,
    normalize_dataset_id,
)

from utils.nav import render_nav
from utils.page import figure_card
from utils.ts_utils import (
    stationarity_metrics_from_series,
    stl_series,
)

ndvi_bp = Blueprint("ndvi", __name__)


def build_series(sub):
    return (
        sub
        .groupby("date")["value"]
        .mean()
        .sort_index()
    )


def _safe_available_indices(dataset_id: str):
    try:
        indices = list_indices(dataset_id=dataset_id)
    except Exception:
        indices = []

    if not indices:
        indices = ["NDVI", "NDMI", "SAVI", "AVI", "EVI", "GNDVI"]

    return indices


def _safe_available_rois(dataset_id: str):
    try:
        rois = get_dataset_rois(dataset_id)
    except Exception:
        rois = ["roi1", "roi2"]

    rois = [str(roi).lower() for roi in rois if str(roi).strip()]

    if not rois:
        rois = ["roi1", "roi2"]

    return sorted(rois)


def build_filters(selected_dataset, selected_index, selected_roi):
    selected_dataset = normalize_dataset_id(selected_dataset)
    available_indices = _safe_available_indices(selected_dataset)
    available_rois = _safe_available_rois(selected_dataset)

    dataset_options = build_dataset_options_html(selected_dataset)

    index_options = ""

    for idx in available_indices:
        selected = "selected" if idx.upper() == selected_index.upper() else ""
        index_options += f"""
        <option value=\"{idx}\" {selected}>
            {idx}
        </option>
        """

    roi_options = ""
    roi_values = list(available_rois)

    if len(available_rois) >= 2:
        roi_values.append("both")

    for roi in roi_values:
        selected = "selected" if roi.lower() == selected_roi.lower() else ""
        label = "Toate ROI-urile" if roi == "both" else roi.upper()
        roi_options += f"""
        <option value=\"{roi}\" {selected}>
            {label}
        </option>
        """

    return f"""

    <form method="get" class="filters-row">

        <label>
            <span class="muted">Dataset</span><br>
            <select name="dataset" class="select-input">
                {dataset_options}
            </select>
        </label>

        <label>
            <span class="muted">Indice</span><br>
            <select name="index" class="select-input">
                {index_options}
            </select>
        </label>

        <label>
            <span class="muted">ROI</span><br>
            <select name="roi" class="select-input">
                {roi_options}
            </select>
        </label>

        <button class="btn btn-primary">
            Aplică
        </button>

    </form>

    """


def get_filtered_dataframe():
    selected_dataset = normalize_dataset_id(
        request.args.get("dataset", DEMO_DATASET_ID)
    )

    available_indices = _safe_available_indices(selected_dataset)

    selected_index = request.args.get("index", "NDVI").upper()

    if selected_index not in available_indices:
        selected_index = available_indices[0]

    available_rois = _safe_available_rois(selected_dataset)

    selected_roi = request.args.get("roi", available_rois[0]).lower()

    if selected_roi != "both" and selected_roi not in available_rois:
        selected_roi = available_rois[0]

    df = load_index_dataframe(
        selected_index,
        dataset_id=selected_dataset,
    )

    if selected_roi.lower() != "both":
        df = df[
            df["roi"].str.lower() == selected_roi.lower()
        ].copy()

    return (
        selected_dataset,
        selected_index,
        selected_roi,
        df,
    )


def get_rois(selected_dataset, selected_roi):
    if selected_roi == "both":
        return _safe_available_rois(selected_dataset)

    return [selected_roi]


def empty_series_message(selected_dataset, selected_index, selected_roi):
    return render_template(
        "base.html",
        title="Date indisponibile",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
            <h1>Date indisponibile</h1>
            <div class="method-box">
                Nu există observații pentru combinația selectată:<br><br>
                Dataset: <strong>{get_dataset_display_name(selected_dataset)}</strong><br>
                Indice: <strong>{selected_index}</strong><br>
                ROI: <strong>{selected_roi.upper()}</strong>
            </div>
        </section>
        """,
    )


@ndvi_bp.route("/spectral")
def spectral():
    (
        selected_dataset,
        selected_index,
        selected_roi,
        df,
    ) = get_filtered_dataframe()

    if df.empty:
        return empty_series_message(selected_dataset, selected_index, selected_roi)

    fig = go.Figure()

    rois = get_rois(selected_dataset, selected_roi)

    for roi in rois:
        sub = df[
            df["roi"].str.lower() == roi.lower()
        ]

        if sub.empty:
            continue

        series = build_series(sub)
        smooth = smooth_series(series)

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines+markers",
                name=f"{roi.upper()} raw"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=smooth.index,
                y=smooth.values,
                mode="lines",
                line=dict(width=4),
                name=f"{roi.upper()} trend"
            )
        )

    fig.update_layout(
        title=f"{selected_index} - {selected_roi}",
        xaxis_title="Data",
        yaxis_title=selected_index,
        hovermode="x unified",
    )

    filters_html = build_filters(
        selected_dataset,
        selected_index,
        selected_roi,
    )

    description = INDEX_DESCRIPTIONS.get(
        selected_index,
        "Indice spectral utilizat în analiza vegetației."
    )

    return render_template(
        "base.html",
        title="Indici spectrali",
        nav_html=render_nav(request.path),
        content=f"""

        <section class="card reveal active">

            <h1>Analiza indicilor spectrali</h1>

            <p class="muted">
                Această secțiune permite explorarea evoluției temporale pentru indicii spectrali.
                Dataseturile încărcate de utilizator sunt integrate în același flux ca ROI-urile demonstrative.
            </p>

            {filters_html}

            <div class="method-box">
                <strong>Dataset selectat:</strong> {get_dataset_display_name(selected_dataset)}<br><br>
                <strong>Index spectral selectat:</strong> {selected_index}<br><br>
                <strong>ROI selectat:</strong> {selected_roi.upper()}<br><br>
                <strong>Descriere:</strong> {description}
            </div>

        </section>

        {figure_card(
            fig,
            f"{selected_index} - {selected_roi}",
            "Serie temporală spectrală",
            section_id="spectral_fig",
            yaxis_title=selected_index,
        )}

        """
    )


@ndvi_bp.route("/decompose")
def decompose():
    (
        selected_dataset,
        selected_index,
        selected_roi,
        df,
    ) = get_filtered_dataframe()

    if df.empty:
        return empty_series_message(selected_dataset, selected_index, selected_roi)

    fig = go.Figure()
    rois = get_rois(selected_dataset, selected_roi)

    for roi in rois:
        sub = df[df["roi"].str.lower() == roi.lower()]
        if sub.empty:
            continue
        series = build_series(sub)
        res = stl_series(series)

        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=f"{roi.upper()} raw"))
        fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend.values, mode="lines", name=f"{roi.upper()} trend"))
        fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal.values, mode="lines", name=f"{roi.upper()} seasonal"))

    fig.update_layout(hovermode="x unified")
    filters_html = build_filters(selected_dataset, selected_index, selected_roi)

    return render_template(
        "base.html",
        title="Decompose",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
            <h1>STL Decomposition</h1>
            <p class="muted">
                STL decomposition separă seria temporală în trend, sezonalitate și reziduuri.
            </p>

            {filters_html}

            <div class="method-box">
                <strong>Dataset:</strong> {get_dataset_display_name(selected_dataset)}<br><br>
                <strong>Interpretare:</strong><br><br>
                Trendul descrie evoluția lentă a vegetației, iar componenta sezonieră evidențiază ciclurile repetitive asociate anotimpurilor.
            </div>
        </section>

        {figure_card(
            fig,
            f"STL - {selected_index}",
            "Descompunere STL",
            section_id="decompose_fig",
            yaxis_title=selected_index,
        )}
        """
    )


@ndvi_bp.route("/trend")
def trend():
    (
        selected_dataset,
        selected_index,
        selected_roi,
        df,
    ) = get_filtered_dataframe()

    if df.empty:
        return empty_series_message(selected_dataset, selected_index, selected_roi)

    fig = go.Figure()
    rois = get_rois(selected_dataset, selected_roi)

    for roi in rois:
        sub = df[df["roi"].str.lower() == roi.lower()]
        if sub.empty:
            continue
        series = build_series(sub)
        res = stl_series(series)
        fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend.values, mode="lines", name=f"{roi.upper()} trend"))

    fig.update_layout(hovermode="x unified")
    filters_html = build_filters(selected_dataset, selected_index, selected_roi)

    return render_template(
        "base.html",
        title="Trend",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
            <h1>Analiza trendului</h1>
            <p class="muted">
                Componenta trend surprinde direcția generală de evoluție a vegetației.
            </p>

            {filters_html}

            <div class="method-box">
                <strong>Dataset:</strong> {get_dataset_display_name(selected_dataset)}<br><br>
                Trendul poate evidenția degradare vegetativă, recuperare graduală sau modificări persistente ale umidității.
            </div>
        </section>

        {figure_card(
            fig,
            f"Trend STL - {selected_index}",
            "Trend extras prin STL",
            section_id="trend_fig",
            yaxis_title=selected_index,
        )}
        """
    )


@ndvi_bp.route("/seasonality")
def seasonality():
    (
        selected_dataset,
        selected_index,
        selected_roi,
        df,
    ) = get_filtered_dataframe()

    if df.empty:
        return empty_series_message(selected_dataset, selected_index, selected_roi)

    fig = go.Figure()
    rois = get_rois(selected_dataset, selected_roi)

    for roi in rois:
        sub = df[df["roi"].str.lower() == roi.lower()]
        if sub.empty:
            continue
        series = build_series(sub)
        res = stl_series(series)
        fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal.values, mode="lines", name=f"{roi.upper()} seasonality"))

    fig.update_layout(hovermode="x unified")
    filters_html = build_filters(selected_dataset, selected_index, selected_roi)

    return render_template(
        "base.html",
        title="Seasonality",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
            <h1>Analiza sezonalității</h1>
            <p class="muted">
                Componenta sezonieră descrie ciclurile repetitive anuale ale vegetației.
            </p>

            {filters_html}

            <div class="method-box">
                <strong>Dataset:</strong> {get_dataset_display_name(selected_dataset)}<br><br>
                Amplitudinea sezonieră indică intensitatea ciclului vegetal.
            </div>
        </section>

        {figure_card(
            fig,
            f"Seasonality STL - {selected_index}",
            "Componentă sezonieră",
            section_id="seasonality_fig",
            yaxis_title="Seasonality",
        )}
        """
    )


@ndvi_bp.route("/anomalies")
def anomalies():
    (
        selected_dataset,
        selected_index,
        selected_roi,
        df,
    ) = get_filtered_dataframe()

    if df.empty:
        return empty_series_message(selected_dataset, selected_index, selected_roi)

    fig = go.Figure()
    rois = get_rois(selected_dataset, selected_roi)
    total_anomalies = 0

    for roi in rois:
        sub = df[df["roi"].str.lower() == roi.lower()]
        if sub.empty:
            continue
        series = build_series(sub)
        res = stl_series(series)
        resid = res.resid.dropna()

        if resid.std(ddof=0) == 0:
            anomalies_idx = []
        else:
            z = (resid - resid.mean()) / resid.std(ddof=0)
            anomalies_idx = z[abs(z) >= 2].index

        total_anomalies += len(anomalies_idx)

        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=f"{roi.upper()} serie"))

        if len(anomalies_idx) > 0:
            vals = series.loc[anomalies_idx]
            fig.add_trace(
                go.Scatter(
                    x=vals.index,
                    y=vals.values,
                    mode="markers",
                    marker=dict(size=11, symbol="diamond"),
                    name=f"{roi.upper()} anomalie"
                )
            )

    fig.update_layout(hovermode="x unified")
    filters_html = build_filters(selected_dataset, selected_index, selected_roi)

    return render_template(
        "base.html",
        title="Anomalies",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
            <h1>Detectarea anomaliilor</h1>
            <p class="muted">
                Anomaliile reprezintă deviații semnificative față de comportamentul temporal normal.
            </p>

            {filters_html}

            <div class="method-box">
                <strong>Dataset:</strong> {get_dataset_display_name(selected_dataset)}<br><br>
                <strong>Număr total anomalii:</strong> {total_anomalies}<br><br>
                Detectarea este realizată pe baza reziduurilor STL și a scorurilor z.
            </div>
        </section>

        {figure_card(
            fig,
            f"Anomalii - {selected_index}",
            "Anomalii STL",
            section_id="anomaly_fig",
            yaxis_title=selected_index,
        )}
        """
    )


@ndvi_bp.route("/stationarity")
def stationarity():
    (
        selected_dataset,
        selected_index,
        selected_roi,
        df,
    ) = get_filtered_dataframe()

    if df.empty:
        return empty_series_message(selected_dataset, selected_index, selected_roi)

    fig = go.Figure()
    metrics_html = ""
    rois = get_rois(selected_dataset, selected_roi)

    for roi in rois:
        sub = df[df["roi"].str.lower() == roi.lower()]
        if sub.empty:
            continue
        series = build_series(sub)
        metrics = stationarity_metrics_from_series(series)

        rolling_mean = series.rolling(window=12, min_periods=3).mean()
        rolling_std = series.rolling(window=12, min_periods=3).std()

        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=f"{roi.upper()} serie"))
        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean.values, mode="lines", name=f"{roi.upper()} mean"))
        fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std.values, mode="lines", name=f"{roi.upper()} std"))

        adf_stat = metrics.get("adf_stat", metrics.get("statistic", "N/A"))
        p_value = metrics.get("p_value", "N/A")
        stationary = metrics.get("stationary", "N/A")

        metrics_html += f"""
        <div class="method-box">
            <strong>{roi.upper()}</strong><br><br>
            ADF statistic: {adf_stat}<br><br>
            p-value: {p_value}<br><br>
            Interpretare: {stationary}
        </div>
        """

    fig.update_layout(hovermode="x unified")
    filters_html = build_filters(selected_dataset, selected_index, selected_roi)

    return render_template(
        "base.html",
        title="Stationarity",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
            <h1>Analiza staționarității</h1>
            <p class="muted">
                Testul ADF verifică dacă seria temporală are proprietăți statistice stabile în timp.
            </p>

            {filters_html}

            <div class="method-box">
                <strong>Dataset selectat:</strong> {get_dataset_display_name(selected_dataset)}
            </div>

            {metrics_html}
        </section>

        {figure_card(
            fig,
            f"Staționaritate - {selected_index}",
            "ADF + rolling statistics",
            section_id="stationarity_fig",
            yaxis_title=selected_index,
        )}
        """
    )
