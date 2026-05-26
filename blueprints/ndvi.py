from flask import Blueprint, render_template, request
import plotly.graph_objects as go

from services.indices_service import (
    load_index_dataframe,
    list_indices,
    INDEX_DESCRIPTIONS,
    smooth_series,
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


def build_filters(selected_index, selected_roi):

    available_indices = list_indices()

    index_options = ""

    for idx in available_indices:

        selected = (
            "selected"
            if idx == selected_index
            else ""
        )

        index_options += f"""
        <option value="{idx}" {selected}>
            {idx}
        </option>
        """

    roi_options = ""

    for roi in ["roi1", "roi2", "both"]:

        selected = (
            "selected"
            if roi == selected_roi
            else ""
        )

        label = (
            "ROI1 + ROI2"
            if roi == "both"
            else roi.upper()
        )

        roi_options += f"""
        <option value="{roi}" {selected}>
            {label}
        </option>
        """

    return f"""

    <form method="get" class="filters-row">

        <select name="index" class="select-input">
            {index_options}
        </select>

        <select name="roi" class="select-input">
            {roi_options}
        </select>

        <button class="btn btn-primary">
            Aplică
        </button>

    </form>

    """


def get_filtered_dataframe():

    selected_index = request.args.get(
        "index",
        "NDVI"
    )

    selected_roi = request.args.get(
        "roi",
        "roi1"
    )

    df = load_index_dataframe(selected_index)

    if selected_roi.lower() != "both":

        df = df[
            df["roi"].str.lower()
            == selected_roi.lower()
        ].copy()

    return (
        selected_index,
        selected_roi,
        df
    )


def get_rois(selected_roi):

    return (
        ["roi1", "roi2"]
        if selected_roi == "both"
        else [selected_roi]
    )

@ndvi_bp.route("/spectral")
def spectral():

    (
        selected_index,
        selected_roi,
        df
    ) = get_filtered_dataframe()

    fig = go.Figure()

    rois = get_rois(selected_roi)

    for roi in rois:

        sub = df[
            df["roi"].str.lower()
            == roi.lower()
        ]

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
        selected_index,
        selected_roi
    )

    description = INDEX_DESCRIPTIONS.get(
        selected_index,
        ""
    )

    return render_template(
        "base.html",
        title="Indici spectrali",
        nav_html=render_nav(request.path),
        content=f"""

        <section class="card reveal active">

            <h1>
                Analiza indicilor spectrali
            </h1>

            <p class="muted">

                Această secțiune permite explorarea
                evoluției temporale pentru indicii
                spectrali calculați din imaginile
                Sentinel-2.

            </p>

            <p class="muted">

                Graficele prezintă atât seria brută,
                cât și trendul temporal estimat.

            </p>

            {filters_html}

            <div class="method-box">

                <strong>
                    Index spectral selectat:
                </strong>

                {selected_index}

                <br><br>

                <strong>
                    ROI selectat:
                </strong>

                {selected_roi.upper()}

                <br><br>

                <strong>
                    Descriere:
                </strong>

                {description}

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
        selected_index,
        selected_roi,
        df
    ) = get_filtered_dataframe()

    fig = go.Figure()

    rois = get_rois(selected_roi)

    for roi in rois:

        sub = df[
            df["roi"].str.lower()
            == roi.lower()
        ]

        series = build_series(sub)

        res = stl_series(series)

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"{roi.upper()} raw"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=res.trend.index,
                y=res.trend.values,
                mode="lines",
                name=f"{roi.upper()} trend"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=res.seasonal.index,
                y=res.seasonal.values,
                mode="lines",
                name=f"{roi.upper()} seasonal"
            )
        )

    fig.update_layout(
        hovermode="x unified"
    )

    filters_html = build_filters(
        selected_index,
        selected_roi
    )

    return render_template(
        "base.html",
        title="Decompose",
        nav_html=render_nav(request.path),
        content=f"""

        <section class="card">

            <h1>
                STL Decomposition
            </h1>

            <p class="muted">

                STL decomposition separă seria temporală
                în trei componente principale:

            </p>

            <ul class="muted">

                <li>
                    trend
                </li>

                <li>
                    sezonalitate
                </li>

                <li>
                    reziduuri
                </li>

            </ul>

            {filters_html}

            <div class="method-box">

                <strong>
                    Interpretare:
                </strong>

                <br><br>

                Trendul descrie evoluția lentă a
                vegetației, iar componenta sezonieră
                evidențiază ciclurile repetitive
                asociate anotimpurilor.

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
        selected_index,
        selected_roi,
        df
    ) = get_filtered_dataframe()

    fig = go.Figure()

    rois = get_rois(selected_roi)

    for roi in rois:

        sub = df[
            df["roi"].str.lower()
            == roi.lower()
        ]

        series = build_series(sub)

        res = stl_series(series)

        fig.add_trace(
            go.Scatter(
                x=res.trend.index,
                y=res.trend.values,
                mode="lines",
                name=f"{roi.upper()} trend"
            )
        )

    fig.update_layout(
        hovermode="x unified"
    )

    filters_html = build_filters(
        selected_index,
        selected_roi
    )

    return render_template(
        "base.html",
        title="Trend",
        nav_html=render_nav(request.path),
        content=f"""

        <section class="card">

            <h1>
                Analiza trendului
            </h1>

            <p class="muted">

                Componenta trend surprinde direcția
                generală de evoluție a vegetației.

            </p>

            <p class="muted">

                Fluctuațiile rapide sunt eliminate,
                fiind păstrată doar variația lentă
                și persistentă.

            </p>

            {filters_html}

            <div class="method-box">

                Trendul poate evidenția:
                <br><br>

                • degradare vegetativă
                <br>

                • recuperare graduală
                <br>

                • schimbări climatice locale
                <br>

                • modificări persistente ale umidității

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
        selected_index,
        selected_roi,
        df
    ) = get_filtered_dataframe()

    fig = go.Figure()

    rois = get_rois(selected_roi)

    for roi in rois:

        sub = df[
            df["roi"].str.lower()
            == roi.lower()
        ]

        series = build_series(sub)

        res = stl_series(series)

        fig.add_trace(
            go.Scatter(
                x=res.seasonal.index,
                y=res.seasonal.values,
                mode="lines",
                name=f"{roi.upper()} seasonality"
            )
        )

    fig.update_layout(
        hovermode="x unified"
    )

    filters_html = build_filters(
        selected_index,
        selected_roi
    )

    return render_template(
        "base.html",
        title="Seasonality",
        nav_html=render_nav(request.path),
        content=f"""

        <section class="card">

            <h1>
                Analiza sezonalității
            </h1>

            <p class="muted">

                Componenta sezonieră descrie
                ciclurile repetitive anuale
                ale vegetației.

            </p>

            <p class="muted">

                Aceasta permite identificarea
                perioadelor de creștere și
                declin vegetativ.

            </p>

            {filters_html}

            <div class="method-box">

                În general:
                <br><br>

                • valorile ridicate apar primăvara și vara
                <br>

                • valorile scăzute apar iarna
                <br>

                • amplitudinea indică intensitatea ciclului vegetal

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
        selected_index,
        selected_roi,
        df
    ) = get_filtered_dataframe()

    fig = go.Figure()

    rois = get_rois(selected_roi)

    total_anomalies = 0

    for roi in rois:

        sub = df[
            df["roi"].str.lower()
            == roi.lower()
        ]

        series = build_series(sub)

        res = stl_series(series)

        resid = res.resid.dropna()

        z = (
            resid - resid.mean()
        ) / resid.std(ddof=0)

        anomalies_idx = z[
            abs(z) >= 2
        ].index

        total_anomalies += len(anomalies_idx)

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"{roi.upper()} serie"
            )
        )

        if len(anomalies_idx) > 0:

            vals = series.loc[anomalies_idx]

            fig.add_trace(
                go.Scatter(
                    x=vals.index,
                    y=vals.values,
                    mode="markers",
                    marker=dict(
                        size=11,
                        symbol="diamond"
                    ),
                    name=f"{roi.upper()} anomalie"
                )
            )

    fig.update_layout(
        hovermode="x unified"
    )

    filters_html = build_filters(
        selected_index,
        selected_roi
    )

    return render_template(
        "base.html",
        title="Anomalies",
        nav_html=render_nav(request.path),
        content=f"""

        <section class="card">

            <h1>
                Detectarea anomaliilor
            </h1>

            <p class="muted">

                Anomaliile reprezintă deviații
                semnificative față de comportamentul
                temporal normal.

            </p>

            <p class="muted">

                Acestea pot indica:
            </p>

            <ul class="muted">

                <li>
                    secetă
                </li>

                <li>
                    stres vegetativ
                </li>

                <li>
                    modificări bruște de umiditate
                </li>

                <li>
                    degradare locală
                </li>

            </ul>

            {filters_html}

            <div class="method-box">

                <strong>
                    Număr total anomalii:
                </strong>

                {total_anomalies}

                <br><br>

                Detectarea este realizată pe baza
                reziduurilor STL și a scorurilor z.

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
        selected_index,
        selected_roi,
        df
    ) = get_filtered_dataframe()

    fig = go.Figure()

    metrics_html = ""

    rois = get_rois(selected_roi)

    for roi in rois:

        sub = df[
            df["roi"].str.lower()
            == roi.lower()
        ]

        series = build_series(sub)

        metrics = stationarity_metrics_from_series(
            series
        )

        rolling_mean = series.rolling(
            window=12,
            min_periods=3
        ).mean()

        rolling_std = series.rolling(
            window=12,
            min_periods=3
        ).std()

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"{roi.upper()} serie"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean.values,
                mode="lines",
                name=f"{roi.upper()} mean"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_std.index,
                y=rolling_std.values,
                mode="lines",
                name=f"{roi.upper()} std"
            )
        )

        adf_stat = metrics.get(
            "adf_stat",
            metrics.get("statistic", "N/A")
        )

        p_value = metrics.get(
            "p_value",
            "N/A"
        )

        stationary = metrics.get(
            "stationary",
            "N/A"
        )

        metrics_html += f"""

        <div class="method-box">

            <strong>
                {roi.upper()}
            </strong>

            <br><br>

            ADF statistic:
            {adf_stat}

            <br><br>

            p-value:
            {p_value}

            <br><br>

            Interpretare:
            {stationary}

        </div>

        """

    fig.update_layout(
        hovermode="x unified"
    )

    filters_html = build_filters(
        selected_index,
        selected_roi
    )

    return render_template(
        "base.html",
        title="Stationarity",
        nav_html=render_nav(request.path),
        content=f"""

        <section class="card">

            <h1>
                Analiza staționarității
            </h1>

            <p class="muted">

                Testul ADF verifică dacă seria
                temporală are proprietăți statistice
                stabile în timp.

            </p>

            <p class="muted">

                Seriile nestaționare prezintă
                trenduri persistente sau variații
                structurale semnificative.

            </p>

            {filters_html}

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