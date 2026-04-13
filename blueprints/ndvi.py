from flask import Blueprint, jsonify, render_template, request
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from services.ndvi_service import load_ndvi, normalize_site_name, pretty_site_name
from utils.nav import render_nav
from utils.page import figure_card
from utils.ts_utils import (
    prepare_monthly_series,
    stationarity_metrics_from_series,
    count_anomalies_in_series,
    stl_series,
)

ndvi_bp = Blueprint("ndvi", __name__)


@ndvi_bp.route("/api/series")
def api_series():
    df = load_ndvi()
    site = request.args.get("site")
    if site:
        site = normalize_site_name(site)
        df = df[df["site"] == site]
    return jsonify(df.to_dict(orient="records"))


@ndvi_bp.route("/toate")
def toate():
    df = load_ndvi().copy()
    df["site_label"] = df["site"].map(pretty_site_name)

    fig = px.line(df, x="date", y="ndvi", color="site_label", markers=True, title="NDVI – toate siturile")

    return render_template(
        "base.html",
        title="NDVI – toate siturile",
        nav_html=render_nav(request.path),
        content=figure_card(
            fig,
            "NDVI – toate siturile",
            "Comparație între Parc București, Agricol Ilfov și Urban Central.",
            section_id="toate_fig",
            yaxis_title="NDVI [0–1]",
        ),
    )


@ndvi_bp.route("/padure")
def padure():
    df = load_ndvi()
    sub = df[df["site"] == "ParcBucuresti"].copy()
    if sub.empty:
        return render_template(
            "base.html",
            title="NDVI – Parc București",
            nav_html=render_nav(request.path),
            content="<section class='card'><h1>NDVI – Parc București</h1><p>Nu există date pentru acest sit.</p></section>",
        )

    fig = px.line(sub, x="date", y="ndvi", markers=True, title="NDVI – Parc București")
    return render_template(
        "base.html",
        title="NDVI – Parc București",
        nav_html=render_nav(request.path),
        content=figure_card(
            fig,
            "NDVI – Parc București",
            "Spațiu verde urban: NDVI relativ ridicat, cu sezonalitate moderată.",
            section_id="padure_fig",
            yaxis_title="NDVI [0–1]",
        ),
    )


@ndvi_bp.route("/agricol")
def agricol():
    df = load_ndvi()
    sub = df[df["site"] == "AgricolIlfov"].copy()
    if sub.empty:
        return render_template(
            "base.html",
            title="NDVI – Agricol Ilfov",
            nav_html=render_nav(request.path),
            content="<section class='card'><h1>NDVI – Agricol Ilfov</h1><p>Nu există date pentru acest sit.</p></section>",
        )

    fig = px.line(sub, x="date", y="ndvi", markers=True, title="NDVI – Agricol Ilfov")
    return render_template(
        "base.html",
        title="NDVI – Agricol Ilfov",
        nav_html=render_nav(request.path),
        content=figure_card(
            fig,
            "NDVI – Agricol Ilfov",
            "Teren agricol: sezonalitate puternică, cu variații clare între iarnă și vară.",
            section_id="agricol_fig",
            yaxis_title="NDVI [0–1]",
        ),
    )


@ndvi_bp.route("/urban")
def urban():
    df = load_ndvi()
    sub = df[df["site"] == "UrbanCentral"].copy()
    if sub.empty:
        return render_template(
            "base.html",
            title="NDVI – Urban Central",
            nav_html=render_nav(request.path),
            content="<section class='card'><h1>NDVI – Urban Central</h1><p>Nu există date pentru acest sit.</p></section>",
        )

    fig = px.line(sub, x="date", y="ndvi", markers=True, title="NDVI – Urban Central")
    return render_template(
        "base.html",
        title="NDVI – Urban Central",
        nav_html=render_nav(request.path),
        content=figure_card(
            fig,
            "NDVI – Urban Central",
            "Zonă urbană densă: NDVI scăzut și variații mai reduse.",
            section_id="urban_fig",
            yaxis_title="NDVI [0–1]",
        ),
    )


@ndvi_bp.route("/stats")
def stats_page():
    df = load_ndvi()
    rows = []

    for site, sub in df.groupby("site"):
        series = prepare_monthly_series(sub)
        rows.append({
            "site": pretty_site_name(site),
            "n_obs": int(len(sub)),
            "date_min": sub["date"].min().strftime("%Y-%m-%d"),
            "date_max": sub["date"].max().strftime("%Y-%m-%d"),
            "mean_ndvi": round(float(sub["ndvi"].mean()), 4),
            "median_ndvi": round(float(sub["ndvi"].median()), 4),
            "min_ndvi": round(float(sub["ndvi"].min()), 4),
            "max_ndvi": round(float(sub["ndvi"].max()), 4),
            "std_ndvi": round(float(sub["ndvi"].std(ddof=0)), 4),
            "amplitude": round(float(sub["ndvi"].max() - sub["ndvi"].min()), 4),
            "anomalies": count_anomalies_in_series(series, period=12),
        })

    stats_df = pd.DataFrame(rows).sort_values("mean_ndvi", ascending=False)

    table_rows = ""
    for _, row in stats_df.iterrows():
        table_rows += f"""
        <tr>
          <td>{row["site"]}</td>
          <td>{row["n_obs"]}</td>
          <td>{row["date_min"]}</td>
          <td>{row["date_max"]}</td>
          <td>{row["mean_ndvi"]}</td>
          <td>{row["median_ndvi"]}</td>
          <td>{row["min_ndvi"]}</td>
          <td>{row["max_ndvi"]}</td>
          <td>{row["std_ndvi"]}</td>
          <td>{row["amplitude"]}</td>
          <td>{row["anomalies"]}</td>
        </tr>
        """

    return render_template(
        "base.html",
        title="Statistici NDVI",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
          <h1>Statistici descriptive NDVI</h1>
          <div class="table-wrap">
            <table class="stats-table">
              <thead>
                <tr>
                  <th>Sit</th>
                  <th>Nr. obs.</th>
                  <th>Data minimă</th>
                  <th>Data maximă</th>
                  <th>Media</th>
                  <th>Mediana</th>
                  <th>Minim</th>
                  <th>Maxim</th>
                  <th>Std. dev.</th>
                  <th>Amplitudine</th>
                  <th>Anomalii</th>
                </tr>
              </thead>
              <tbody>{table_rows}</tbody>
            </table>
          </div>
        </section>
        """,
    )


@ndvi_bp.route("/compare-metrics")
def compare_metrics_page():
    df = load_ndvi()
    rows = []

    for site, sub in df.groupby("site"):
        series = prepare_monthly_series(sub)
        station = stationarity_metrics_from_series(series)

        rows.append({
            "site": pretty_site_name(site),
            "mean_ndvi": round(float(sub["ndvi"].mean()), 4),
            "std_ndvi": round(float(sub["ndvi"].std(ddof=0)), 4),
            "amplitude": round(float(sub["ndvi"].max() - sub["ndvi"].min()), 4),
            "anomalies": count_anomalies_in_series(series, period=12),
            "adf_pvalue": None if station["p_value"] is None else round(station["p_value"], 4),
            "stationarity": station["stationary"],
        })

    cmp_df = pd.DataFrame(rows)
    cmp_df["adf_pvalue"] = cmp_df["adf_pvalue"].fillna(np.nan)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cmp_df["site"], y=cmp_df["mean_ndvi"], name="Media NDVI"))
    fig.add_trace(go.Bar(x=cmp_df["site"], y=cmp_df["amplitude"], name="Amplitudine"))
    fig.add_trace(go.Bar(x=cmp_df["site"], y=cmp_df["std_ndvi"], name="Std. dev."))

    table_rows = ""
    for _, row in cmp_df.iterrows():
        pval = "n/a" if pd.isna(row["adf_pvalue"]) else row["adf_pvalue"]
        table_rows += f"""
        <tr>
          <td>{row["site"]}</td>
          <td>{row["mean_ndvi"]}</td>
          <td>{row["std_ndvi"]}</td>
          <td>{row["amplitude"]}</td>
          <td>{row["anomalies"]}</td>
          <td>{pval}</td>
          <td>{row["stationarity"]}</td>
        </tr>
        """

    content = f"""
    <section class="card">
      <h1>Compare Metrics</h1>
      <div class="table-wrap">
        <table class="stats-table">
          <thead>
            <tr>
              <th>Sit</th>
              <th>Media NDVI</th>
              <th>Std. dev.</th>
              <th>Amplitudine</th>
              <th>Anomalii</th>
              <th>ADF p-value</th>
              <th>Staționaritate</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>
      </div>
    </section>
    {figure_card(fig, "Compare Metrics – situri NDVI", "Comparație între indicatorii principali ai seriilor NDVI.", section_id="compare_metrics_fig", yaxis_title="Valoare indicator")}
    """

    return render_template(
        "base.html",
        title="Compare Metrics",
        nav_html=render_nav(request.path),
        content=content,
    )


@ndvi_bp.route("/stationarity")
def stationarity_page():
    df = load_ndvi()
    rows = []
    sections = []

    for idx, (site, sub) in enumerate(df.groupby("site"), start=1):
        series = prepare_monthly_series(sub)
        metrics = stationarity_metrics_from_series(series)

        rolling_mean = series.rolling(window=12, min_periods=3).mean()
        rolling_std = series.rolling(window=12, min_periods=3).std()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name="Seria NDVI"))
        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean.values, mode="lines", name="Media mobilă (12)"))
        fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std.values, mode="lines", name="Std. mobilă (12)"))

        sections.append(
            figure_card(
                fig,
                f"Staționaritate – {pretty_site_name(site)}",
                (
                    f"Test ADF: {'n/a' if metrics['adf_stat'] is None else round(metrics['adf_stat'], 4)} | "
                    f"p-value: {'n/a' if metrics['p_value'] is None else round(metrics['p_value'], 4)} | "
                    f"Interpretare: {metrics['stationary']}"
                ),
                section_id=f"stationarity_{idx}",
                yaxis_title="NDVI [0–1]",
            )
        )

        rows.append({
            "site": pretty_site_name(site),
            "adf_stat": "n/a" if metrics["adf_stat"] is None else round(metrics["adf_stat"], 4),
            "p_value": "n/a" if metrics["p_value"] is None else round(metrics["p_value"], 4),
            "lags_used": "n/a" if metrics["lags_used"] is None else metrics["lags_used"],
            "n_obs": metrics["n_obs"],
            "interpretation": metrics["stationary"],
        })

    table_rows = ""
    for row in rows:
        table_rows += f"""
        <tr>
          <td>{row["site"]}</td>
          <td>{row["adf_stat"]}</td>
          <td>{row["p_value"]}</td>
          <td>{row["lags_used"]}</td>
          <td>{row["n_obs"]}</td>
          <td>{row["interpretation"]}</td>
        </tr>
        """

    content = f"""
    <section class="card">
      <h1>Analiza staționarității</h1>
      <div class="table-wrap">
        <table class="stats-table">
          <thead>
            <tr>
              <th>Sit</th>
              <th>ADF statistic</th>
              <th>p-value</th>
              <th>Lags folosite</th>
              <th>Nr. obs.</th>
              <th>Interpretare</th>
            </tr>
          </thead>
          <tbody>{table_rows}</tbody>
        </table>
      </div>
    </section>
    {''.join(sections)}
    """

    return render_template(
        "base.html",
        title="Staționaritate NDVI",
        nav_html=render_nav(request.path),
        content=content,
    )


@ndvi_bp.route("/decompose")
def decompose_page():
    df = load_ndvi()
    selected_site = request.args.get("site", "AgricolIlfov")
    selected_site = normalize_site_name(selected_site)

    sub = df[df["site"] == selected_site].copy()
    if sub.empty:
        return render_template(
            "base.html",
            title="Decompose",
            nav_html=render_nav(request.path),
            content="<section class='card'><h1>Decompose</h1><p>Nu există date pentru situl selectat.</p></section>",
        )

    series = prepare_monthly_series(sub)
    res = stl_series(series)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Serie originală"))
    fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend.values, mode="lines", name="Trend"))
    fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal.values, mode="lines", name="Sezonalitate"))
    fig.add_trace(go.Scatter(x=res.resid.index, y=res.resid.values, mode="lines", name="Reziduu"))

    site_links = "".join(
        f"<a class='btn-link secondary' href='/decompose?site={site}'>{pretty_site_name(site)}</a>"
        for site in sorted(df["site"].unique())
    )

    return render_template(
        "base.html",
        title="Decompose",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
          <h1>STL Decompose – {pretty_site_name(selected_site)}</h1>
          <p class="muted">
            Pagina afișează descompunerea STL pentru un sit ales: seria originală, trendul,
            componenta sezonieră și reziduul.
          </p>
          <div class="roi-actions">{site_links}</div>
        </section>
        {figure_card(fig, f"Decompose – {pretty_site_name(selected_site)}", "Descompunere STL a seriei lunare NDVI.", section_id="decompose_fig", yaxis_title="NDVI [0–1]")}
        """,
    )


@ndvi_bp.route("/trend")
def trend_all():
    df = load_ndvi()
    fig = go.Figure()

    for site, sub in df.groupby("site"):
        try:
            series = prepare_monthly_series(sub)
            res = stl_series(series)
            fig.add_trace(go.Scatter(
                x=res.trend.index,
                y=res.trend.values,
                mode="lines",
                name=f"Trend – {pretty_site_name(site)}",
            ))
        except Exception:
            pass

    return render_template(
        "base.html",
        title="Trend (STL)",
        nav_html=render_nav(request.path),
        content=figure_card(
            fig,
            "Trend (STL) – toate siturile",
            "Componenta de trend extrasă prin STL pentru toate siturile.",
            section_id="trend_fig",
            yaxis_title="NDVI [0–1]",
        ),
    )


@ndvi_bp.route("/seasonality")
def seasonality_all():
    df = load_ndvi()
    fig = go.Figure()

    for site, sub in df.groupby("site"):
        try:
            series = prepare_monthly_series(sub)
            res = stl_series(series)
            fig.add_trace(go.Scatter(
                x=res.seasonal.index,
                y=res.seasonal.values,
                mode="lines",
                name=f"Sezonalitate – {pretty_site_name(site)}",
            ))
        except Exception:
            pass

    return render_template(
        "base.html",
        title="Sezonalitate (STL)",
        nav_html=render_nav(request.path),
        content=figure_card(
            fig,
            "Sezonalitate (STL) – toate siturile",
            "Componenta sezonieră anuală extrasă prin STL pentru toate siturile.",
            section_id="seasonality_fig",
            yaxis_title="Componentă sezonieră",
        ),
    )


@ndvi_bp.route("/anomalies")
def anomalies_all():
    df = load_ndvi().copy()
    df["site_label"] = df["site"].map(pretty_site_name)

    fig = px.line(
        df,
        x="date",
        y="ndvi",
        color="site_label",
        markers=True,
        title="NDVI + Anomalii – toate siturile",
    )

    for site, sub in df.groupby("site"):
        try:
            series = prepare_monthly_series(sub)
            res = stl_series(series)
            resid = res.resid.dropna()
            std = resid.std(ddof=0)

            if std == 0 or pd.isna(std):
                continue

            z = (resid - resid.mean()) / std
            idx = z[abs(z) >= 2].index

            if len(idx) > 0:
                vals = series.loc[idx]
                fig.add_scatter(
                    x=vals.index,
                    y=vals.values,
                    mode="markers",
                    marker=dict(size=11, symbol="diamond"),
                    name=f"Anomalii – {pretty_site_name(site)}",
                )
        except Exception:
            pass

    return render_template(
        "base.html",
        title="Anomalii",
        nav_html=render_nav(request.path),
        content=figure_card(
            fig,
            "Anomalii (STL, |z| ≥ 2) – toate siturile",
            "Anomaliile sunt puncte cu reziduuri neobișnuite după decompoziția STL.",
            section_id="anomalies_fig",
            yaxis_title="NDVI [0–1]",
        ),
    )