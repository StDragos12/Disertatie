from flask import Blueprint, render_template, request
import plotly.graph_objects as go
from utils.insights import generate_insights

from services.synthetic_service import (
    SYNTHETIC_SERIES_META,
    generate_synthetic_series,
    generate_temperature_demo_series,
)
from utils.nav import render_nav
from utils.page import figure_card
from utils.ts_utils import stationarity_metrics_from_series, count_anomalies_in_series, stl_series

synthetic_bp = Blueprint("synthetic", __name__)


def generic_series_page(title: str, df_series, current_path: str, value_label: str = "Valoare", series_kind: str = "generic"):
    series = df_series.set_index("date")["value"].asfreq("MS")
    station = stationarity_metrics_from_series(series)
    anomaly_count = count_anomalies_in_series(series, period=12)

    insights = generate_insights(series)

    insights_html = "<ul class='insights-list'>"
    for insight in insights:
        insights_html += f"<li>{insight}</li>"
    insights_html += "</ul>"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_series["date"],
        y=df_series["value"],
        mode="lines+markers",
        name=df_series["series_name"].iloc[0],
    ))

    extra_sections = ""
    try:
        res = stl_series(series, period=12)
        fig_components = go.Figure()
        fig_components.add_trace(go.Scatter(x=res.trend.index, y=res.trend.values, mode="lines", name="Trend"))
        fig_components.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal.values, mode="lines", name="Sezonalitate"))
        fig_components.add_trace(go.Scatter(x=res.resid.index, y=res.resid.values, mode="lines", name="Reziduu"))

        extra_sections = figure_card(
            fig_components,
            f"Componente STL – {title}",
            "Descompunere STL pentru evidențierea trendului, sezonalității și reziduurilor.",
            section_id=f"{series_kind}_components",
            yaxis_title=value_label,
        )
    except Exception:
        extra_sections = ""

    adf_text = "n/a" if station["adf_stat"] is None else round(station["adf_stat"], 4)
    p_text = "n/a" if station["p_value"] is None else round(station["p_value"], 4)

    content = f"""
<section class="card">
  <h1>{title}</h1>
  <p class="muted">{df_series["description"].iloc[0]}</p>

  <div class="method-box">
    <strong>Categoria:</strong> {df_series["category"].iloc[0]}<br>
    <strong>Număr observații:</strong> {len(df_series)}<br>
    <strong>ADF statistic:</strong> {adf_text}<br>
    <strong>p-value:</strong> {p_text}<br>
    <strong>Interpretare:</strong> {station["stationary"]}<br>
    <strong>Anomalii STL:</strong> {anomaly_count}
  </div>
</section>

<section class="card">
  <h2>Interpretare automată</h2>
  {insights_html}
</section>

{figure_card(fig, f"Serie temporală – {title}", "Vizualizarea seriei.", section_id=f"{series_kind}_main", yaxis_title=value_label)}

{extra_sections}
"""

    return render_template(
        "base.html",
        title=title,
        nav_html=render_nav(current_path),
        content=content,
    )


@synthetic_bp.route("/synthetic")
def synthetic_index():
    cards = ""
    for key, meta in SYNTHETIC_SERIES_META.items():
        cards += f"""
        <div class="roi-card">
          <h2>{meta["title"]}</h2>
          <div class="badge">{meta["category"]}</div>
          <div class="roi-meta">{meta["description"]}</div>
          <div class="roi-actions">
            <a class="btn-link" href="/synthetic/{key}">Deschide seria</a>
          </div>
        </div>
        """

    return render_template(
        "base.html",
        title="Serii sintetice",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
          <h1>Serii sintetice</h1>
          <div class="roi-grid">{cards}</div>
        </section>
        """,
    )


@synthetic_bp.route("/synthetic/<series_key>")
def synthetic_series_page(series_key: str):
    df_series = generate_synthetic_series(series_key)
    return generic_series_page(
        title=df_series["series_name"].iloc[0],
        df_series=df_series,
        current_path=request.path,
        value_label="Valoare",
        series_kind=f"synthetic_{series_key.replace('-', '_')}",
    )


@synthetic_bp.route("/temperature-demo")
def temperature_demo_page():
    df_series = generate_temperature_demo_series()
    return generic_series_page(
        title="Temperatură lunară demonstrativă",
        df_series=df_series,
        current_path=request.path,
        value_label="Temperatură [°C]",
        series_kind="temperature_demo",
    )