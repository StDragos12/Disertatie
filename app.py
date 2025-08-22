from flask import Flask, jsonify, request, render_template
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import STL

app = Flask(__name__)

# --- Config & data ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ndvi_timeseries_csv_multi.csv"  # coloane: date,site,ndvi

def load_ndvi():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
    df = df.dropna(subset=["date", "site", "ndvi"]).sort_values(["site", "date"])
    return df

def get_sites(df):
    return sorted(df["site"].unique().tolist())

# --- layout helpers ---
def shell_html(title: str, body_html: str) -> str:
    nav = (
        "<div class='header'><div class='container'><div class='nav'>"
        "<a class='brand' href='/'>NDVI Viz</a>"
        "<div class='nav-links'>"
        "<a href='/toate'>Compară toate</a>"
        "<a href='/padure'>Pădure</a>"
        "<a href='/agricol'>Agricol</a>"
        "<a href='/urban'>Urban</a>"
        "<a href='/trend'>Trend</a>"
        "<a href='/seasonality'>Sezonalitate</a>"
        "<a href='/anomalies'>Anomalii</a>"
        "</div></div></div></div>"
    )
    return f"""<!doctype html><html><head>
      <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{title}</title>
      <link rel="stylesheet" href="/static/styles.css">
      <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    </head><body>
      {nav}
      <main class="container">{body_html}</main>
    </body></html>"""

def fig_page(fig: go.Figure, title: str, intro: str = "") -> str:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="",
        yaxis_title="NDVI [0–1]"
    )
    intro_html = f"<p class='muted'>{intro}</p>" if intro else ""
    return shell_html(
        title,
        f"<section class='card'><h1>{title}</h1>{intro_html}<div id='fig'></div></section>"
        f"<script>const spec={fig.to_json()};"
        "Plotly.newPlot('fig', spec.data, spec.layout, {responsive:true});</script>"
    )

# --- homepage ---
@app.route("/")
def home():
    df = load_ndvi()
    sites = get_sites(df)
    return render_template("index.html", sites=sites)

# --- API pentru date brute ---
@app.route("/api/series")
def api_series():
    df = load_ndvi()
    site = request.args.get("site")
    if site:
        df = df[df["site"] == site]
    return jsonify(df.to_dict(orient="records"))

# --- pagini de vizualizare ---
@app.route("/toate")
def toate():
    df = load_ndvi()
    fig = px.line(df, x="date", y="ndvi", color="site", title="NDVI – toate siturile")
    intro = "Comparație între Pădure, Teren Agricol și Urban (Serii lunare Sentinel-2 / NDVI)."
    return fig_page(fig, "NDVI – toate siturile", intro)

@app.route("/padure")
def padure():
    df = load_ndvi()
    sub = df[df["site"] == "Padure"]
    fig = px.line(sub, x="date", y="ndvi", title="NDVI – Pădure")
    intro = "Ecosistem forestier: NDVI ridicat și relativ stabil, sezonalitate moderată."
    return fig_page(fig, "NDVI – Pădure", intro)

@app.route("/agricol")
def agricol():
    df = load_ndvi()
    sub = df[df["site"] == "TerenAgricol"]
    fig = px.line(sub, x="date", y="ndvi", title="NDVI – Teren Agricol")
    intro = "Teren agricol: sezonalitate puternică (minim iarna, vârf vara)."
    return fig_page(fig, "NDVI – Teren Agricol", intro)

@app.route("/urban")
def urban():
    df = load_ndvi()
    sub = df[df["site"] == "Urban"]
    fig = px.line(sub, x="date", y="ndvi", title="NDVI – Urban")
    intro = "Zonă urbană: NDVI mic, aproape plat – puțină vegetație."
    return fig_page(fig, "NDVI – Urban", intro)

# --- STL helpers + analize ---
def stl_series(series):
    s = series.asfreq("MS")
    return STL(s, period=12, robust=True).fit()

@app.route("/trend")
def trend_all():
    df = load_ndvi()
    fig = go.Figure()
    for site, sub in df.groupby("site"):
        res = stl_series(sub.set_index("date")["ndvi"])
        fig.add_trace(go.Scatter(x=sub["date"], y=res.trend, name=f"Trend – {site}"))
    return fig_page(fig, "Trend (STL) – toate siturile",
                    "Componenta de trend obținută prin STL (perioadă 12 luni).")

@app.route("/seasonality")
def seasonality_all():
    df = load_ndvi()
    fig = go.Figure()
    for site, sub in df.groupby("site"):
        res = stl_series(sub.set_index("date")["ndvi"])
        fig.add_trace(go.Scatter(x=sub["date"], y=res.seasonal, name=f"Sezonalitate – {site}"))
    return fig_page(fig, "Sezonalitate (STL) – toate siturile",
                    "Componenta sezonieră anuală (periodicitate 12 luni).")

@app.route("/anomalies")
def anomalies_all():
    df = load_ndvi()
    fig = px.line(df, x="date", y="ndvi", color="site", title="NDVI + Anomalii – toate siturile")
    for site, sub in df.groupby("site"):
        res = stl_series(sub.set_index("date")["ndvi"])
        resid = res.resid.dropna()
        z = (resid - resid.mean()) / resid.std(ddof=0)
        idx = z[abs(z) >= 2].index
        if len(idx) > 0:
            vals = sub.set_index("date").loc[idx, "ndvi"]
            fig.add_scatter(
                x=vals.index, y=vals.values, mode="markers",
                marker=dict(size=9, symbol="diamond"),
                name=f"Anomalii – {site}"
            )
    return fig_page(fig, "Anomalii (STL, |z|≥2) – toate siturile",
                    "Anomalii = valori neobișnuite ale reziduurilor după STL (|z|≥2).")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
