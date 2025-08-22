from flask import Flask, jsonify, request
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

# --- HTML helpers ---
BASE_CSS = """
:root{--bg:#0b1220;--card:#111a2b;--text:#e7eefc;--muted:#9fb0cf;--accent:#6aa9ff;--accent2:#ffd166;}
*{box-sizing:border-box} html,body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,Arial}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.container{width:min(1100px,92%);margin:0 auto}
.header{position:sticky;top:0;background:rgba(11,18,32,.85);backdrop-filter:blur(6px);border-bottom:1px solid #1f2a44}
.nav{display:flex;align-items:center;justify-content:space-between;padding:12px 0}
.brand{font-weight:800;letter-spacing:.4px}
.nav a{margin-left:14px}
.card{background:var(--card);border:1px solid #1f2a44;border-radius:14px;padding:14px 16px;margin-top:14px}
select{background:#0f1524;border:1px solid #2a3b63;color:var(--text);border-radius:8px;padding:8px}
"""

def shell_html(title:str, inner_html:str) -> str:
    nav = (
        "<div class='header'><div class='container'><div class='nav'>"
        "<div class='brand'>NDVI Viz</div>"
        "<div>"
        "<a href='/demo'>Demo</a>"
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
      <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
      <style>{BASE_CSS}</style>
    </head>
    <body>
      {nav}
      <main class="container">{inner_html}</main>
    </body></html>"""

def fig_page(fig:go.Figure, title:str, extra_top:str="") -> str:
    fig.update_layout(template="plotly_white", margin=dict(l=20,r=20,t=50,b=20),
                      xaxis_title="", yaxis_title="NDVI [0–1]")
    return shell_html(
        title,
        f"<div class='card'><h2 style='margin:6px 0 10px'>{title}</h2>{extra_top}<div id='fig'></div></div>"
        f"<script>const spec = {fig.to_json()};"
        "Plotly.newPlot('fig', spec.data, spec.layout, {responsive:true});</script>"
    )

# --- API pentru demo ---
@app.route("/api/series")
def api_series():
    df = load_ndvi()
    site = request.args.get("site")
    if site and site != "Toate":
        df = df[df["site"] == site]
    return jsonify(df.to_dict(orient="records"))

# --- Pagini ---
@app.route("/")
def home():
    return shell_html("NDVI Viz", "<div class='card'><h2>Bun venit!</h2><p>Mergi la <a href='/demo'>Demo</a>.</p></div>")

@app.route("/demo")
def demo():
    df = load_ndvi()
    sites = ["Toate"] + get_sites(df)
    options_html = "".join([f"<option value='{s}'>{s}</option>" for s in sites])

    dropdown = (
        "<div style='margin-bottom:10px'>"
        "<label for='siteSel' style='margin-right:8px;color:var(--muted)'>Alege situl:</label>"
        f"<select id='siteSel'>{options_html}</select>"
        "</div>"
    )

    fig = go.Figure()
    page = fig_page(fig, "NDVI – demo interactiv", dropdown)

    js = """
    <script>
      const sel = document.getElementById('siteSel');
      async function refresh(){
        const site = sel.value;
        const url = site==='Toate' ? '/api/series' : '/api/series?site=' + encodeURIComponent(site);
        const data = await (await fetch(url)).json();
        const bySite = {};
        for(const d of data){
          if(!bySite[d.site]) bySite[d.site] = {x:[], y:[]};
          bySite[d.site].x.push(d.date);
          bySite[d.site].y.push(d.ndvi);
        }
        const traces = Object.entries(bySite).map(([name,arr]) => ({
          x: arr.x, y: arr.y, mode:'lines', name
        }));
        const layout = { title: 'NDVI – ' + site, template:'plotly_white',
                         margin:{l:20,r:20,t:50,b:20}, yaxis:{title:'NDVI [0–1]'} };
        Plotly.newPlot('fig', traces, layout, {responsive:true});
      }
      sel.addEventListener('change', refresh);
      refresh();
    </script>
    """
    return page.replace("</body></html>", js + "</body></html>")

@app.route("/padure")
def padure():
    df = load_ndvi()
    sub = df[df["site"] == "Padure"]
    fig = px.line(sub, x="date", y="ndvi", title="NDVI – Pădure")
    return fig_page(fig, "NDVI – Pădure")

@app.route("/agricol")
def agricol():
    df = load_ndvi()
    sub = df[df["site"] == "TerenAgricol"]
    fig = px.line(sub, x="date", y="ndvi", title="NDVI – Teren Agricol")
    return fig_page(fig, "NDVI – Teren Agricol")

@app.route("/urban")
def urban():
    df = load_ndvi()
    sub = df[df["site"] == "Urban"]
    fig = px.line(sub, x="date", y="ndvi", title="NDVI – Urban")
    return fig_page(fig, "NDVI – Urban")

# --- STL utils ---
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
    return fig_page(fig, "Trend (STL) – toate siturile")

@app.route("/seasonality")
def seasonality_all():
    df = load_ndvi()
    fig = go.Figure()
    for site, sub in df.groupby("site"):
        res = stl_series(sub.set_index("date")["ndvi"])
        fig.add_trace(go.Scatter(x=sub["date"], y=res.seasonal, name=f"Sezonalitate – {site}"))
    return fig_page(fig, "Sezonalitate (STL) – toate siturile")

@app.route("/anomalies")
def anomalies_all():
    df = load_ndvi()
    fig = px.line(df, x="date", y="ndvi", color="site", title="NDVI + anomalii – toate siturile")
    for site, sub in df.groupby("site"):
        res = stl_series(sub.set_index("date")["ndvi"])
        resid = res.resid.dropna()
        z = (resid - resid.mean()) / resid.std(ddof=0)
        idx = z[abs(z) >= 2].index
        if len(idx) > 0:
            vals = sub.set_index("date").loc[idx, "ndvi"]
            fig.add_scatter(x=vals.index, y=vals.values, mode="markers",
                            marker=dict(size=9, symbol="diamond"),
                            name=f"Anomalii – {site}")
    return fig_page(fig, "Anomalii (STL, |z|≥2) – toate siturile")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
