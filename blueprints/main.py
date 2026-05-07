from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from services.indices_service import (
    list_indices,
    load_index_dataframe,
    build_indices_wide_dataframe,
    INDEX_DESCRIPTIONS,
)
from config import HOME_SECTIONS, ROI_INFO
from services.ndvi_service import load_ndvi, get_sites, pretty_site_name
from services.synthetic_service import generate_synthetic_series, generate_temperature_demo_series
from utils.nav import render_nav
from utils.page import figure_card
from utils.ts_utils import (
    prepare_monthly_series,
    stationarity_metrics_from_series,
    count_anomalies_in_series,
    extract_features,
    classify_series_features,
    pairwise_dtw_matrix,
)

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    df = load_ndvi()
    sites = [pretty_site_name(s) for s in get_sites(df)]

    return render_template(
        "index.html",
        nav_html=render_nav(request.path),
        sites=sites,
        home_sections=HOME_SECTIONS,
    )


@main_bp.route("/series-catalog")
def series_catalog():
    return render_template(
        "base.html",
        title="Catalog serii",
        nav_html=render_nav(request.path),
        content="""
        <section class="card reveal active">
            <h1>Catalog de serii temporale</h1>
            <p class="muted">
                Seriile sunt organizate în funcție de proprietățile lor statistice:
                <strong>staționaritate</strong>, <strong>sezonalitate</strong> și
                <strong>tipul datelor</strong> (sintetice, climatice sau NDVI reale).
            </p>
        </section>

        <div class="catalog-grid">

            <div class="catalog-card">
                <h2>Staționare</h2>
                <p class="muted">Serii fără trend și fără sezonalitate clară.</p>
                <ul>
                    <li><a href="/synthetic/white-noise">White Noise</a></li>
                </ul>
            </div>

            <div class="catalog-card">
                <h2>Nestaționare</h2>
                <p class="muted">Serii cu trend sau evoluție structurală în timp.</p>
                <ul>
                    <li><a href="/synthetic/random-walk">Random Walk</a></li>
                    <li><a href="/synthetic/linear-trend">Trend liniar + zgomot</a></li>
                </ul>
            </div>

            <div class="catalog-card">
                <h2>Cu sezonalitate</h2>
                <p class="muted">Serii cu pattern periodic sau ciclu repetitiv.</p>
                <ul>
                    <li><a href="/synthetic/seasonal-noise">Sinusoidală + zgomot</a></li>
                    <li><a href="/synthetic/trend-seasonal">Trend + sezonalitate</a></li>
                    <li><a href="/temperature-demo">Temperatură demonstrativă</a></li>
                    <li><a href="/agricol">NDVI agricol</a></li>
                </ul>
            </div>

            <div class="catalog-card">
                <h2>Fără sezonalitate clară</h2>
                <p class="muted">Serii fără ciclu periodic evident.</p>
                <ul>
                    <li><a href="/urban">NDVI urban</a></li>
                    <li><a href="/padure">NDVI parc</a></li>
                </ul>
            </div>

        </div>
        """,
    )


def _series_summary_row(series_name: str, category: str, series: pd.Series) -> dict:
    station = stationarity_metrics_from_series(series)
    anomalies = count_anomalies_in_series(series, period=12)

    return {
        "series_name": series_name,
        "category": category,
        "mean": round(float(series.mean()), 4),
        "std": round(float(series.std(ddof=0)), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "amplitude": round(float(series.max() - series.min()), 4),
        "adf_pvalue": None if station["p_value"] is None else round(float(station["p_value"]), 4),
        "stationarity": station["stationary"],
        "anomalies": anomalies,
    }


@main_bp.route("/compare-series")
def compare_series_page():
    rows = []
    fig = go.Figure()

    white_df = generate_synthetic_series("white-noise")
    white_series = white_df.set_index("date")["value"].asfreq("MS")
    rows.append(_series_summary_row("White Noise", "Sintetică / staționară", white_series))
    fig.add_trace(go.Scatter(
        x=white_series.index,
        y=white_series.values,
        mode="lines",
        name="White Noise",
    ))

    rw_df = generate_synthetic_series("random-walk")
    rw_series = rw_df.set_index("date")["value"].asfreq("MS")
    rows.append(_series_summary_row("Random Walk", "Sintetică / nestaționară", rw_series))
    fig.add_trace(go.Scatter(
        x=rw_series.index,
        y=rw_series.values,
        mode="lines",
        name="Random Walk",
    ))

    temp_df = generate_temperature_demo_series()
    temp_series = temp_df.set_index("date")["value"].asfreq("MS")
    rows.append(_series_summary_row("Temperatură demonstrativă", "Climatică / sezonieră", temp_series))
    fig.add_trace(go.Scatter(
        x=temp_series.index,
        y=temp_series.values,
        mode="lines",
        name="Temperatură demonstrativă",
    ))

    df = load_ndvi()
    for site_code, sub in df.groupby("site"):
        series = prepare_monthly_series(sub)
        pretty_name = pretty_site_name(site_code)
        rows.append(_series_summary_row(pretty_name, "NDVI / reală", series))
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=pretty_name,
        ))

    cmp_df = pd.DataFrame(rows)

    table_rows = ""
    for _, row in cmp_df.iterrows():
        pval = "n/a" if pd.isna(row["adf_pvalue"]) else row["adf_pvalue"]
        table_rows += f"""
        <tr>
          <td>{row["series_name"]}</td>
          <td>{row["category"]}</td>
          <td>{row["mean"]}</td>
          <td>{row["std"]}</td>
          <td>{row["min"]}</td>
          <td>{row["max"]}</td>
          <td>{row["amplitude"]}</td>
          <td>{pval}</td>
          <td>{row["stationarity"]}</td>
          <td>{row["anomalies"]}</td>
        </tr>
        """

    return render_template(
        "base.html",
        title="Compare Series",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
          <h1>Comparare tipuri de serii</h1>
          <p class="muted">
            Această pagină compară într-un singur loc serii sintetice, serii climatice demonstrative
            și seriile NDVI reale, pentru a evidenția diferențele dintre
            <strong>staționaritate</strong>, <strong>trend</strong>, <strong>sezonalitate</strong>
            și <strong>anomalii</strong>.
          </p>
          <div class="method-box">
            <strong>Interpretare:</strong><br>
            White Noise este un exemplu de serie staționară, Random Walk este un exemplu clasic
            de serie nestaționară, temperatura demonstrativă ilustrează sezonalitatea, iar seriile
            NDVI evidențiază comportamente reale din date satelitare.
          </div>
        </section>

        <section class="card reveal active">
          <h1>Tabel comparativ</h1>
          <div class="table-wrap">
            <table class="stats-table">
              <thead>
                <tr>
                  <th>Serie</th>
                  <th>Categorie</th>
                  <th>Media</th>
                  <th>Std. dev.</th>
                  <th>Minim</th>
                  <th>Maxim</th>
                  <th>Amplitudine</th>
                  <th>ADF p-value</th>
                  <th>Staționaritate</th>
                  <th>Anomalii</th>
                </tr>
              </thead>
              <tbody>
                {table_rows}
              </tbody>
            </table>
          </div>
        </section>

        {figure_card(
            fig,
            "Vizualizare comparativă a seriilor",
            "Grafic comparativ între seriile sintetice, demonstrative și NDVI.",
            section_id="compare_series_fig",
            yaxis_title="Valoare",
        )}
        """,
    )

@main_bp.route("/comparative-analysis")
def comparative_analysis_page():
    df = load_ndvi()

    white_df = generate_synthetic_series("white-noise")
    white_series = white_df.set_index("date")["value"].asfreq("MS")

    rw_df = generate_synthetic_series("random-walk")
    rw_series = rw_df.set_index("date")["value"].asfreq("MS")

    temp_df = generate_temperature_demo_series()
    temp_series = temp_df.set_index("date")["value"].asfreq("MS")

    agricol_series = prepare_monthly_series(df[df["site"] == "AgricolIlfov"].copy())
    urban_series = prepare_monthly_series(df[df["site"] == "UrbanCentral"].copy())
    parc_series = prepare_monthly_series(df[df["site"] == "ParcBucuresti"].copy())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=white_series.index, y=white_series.values, mode="lines", name="White Noise"
    ))
    fig.add_trace(go.Scatter(
        x=rw_series.index, y=rw_series.values, mode="lines", name="Random Walk"
    ))
    fig.add_trace(go.Scatter(
        x=temp_series.index, y=temp_series.values, mode="lines", name="Temperatură demonstrativă"
    ))
    fig.add_trace(go.Scatter(
        x=agricol_series.index, y=agricol_series.values, mode="lines", name="NDVI Agricol"
    ))
    fig.add_trace(go.Scatter(
        x=urban_series.index, y=urban_series.values, mode="lines", name="NDVI Urban"
    ))
    fig.add_trace(go.Scatter(
        x=parc_series.index, y=parc_series.values, mode="lines", name="NDVI Parc"
    ))

    return render_template(
        "base.html",
        title="Analiză comparativă",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h1>Analiză comparativă interpretativă</h1>
            <p class="muted">
                Această pagină sintetizează diferențele conceptuale dintre seriile analizate,
                pentru a evidenția proprietăți precum staționaritatea, sezonalitatea,
                variabilitatea și comportamentul specific datelor reale versus sintetice.
            </p>
        </section>

        <section class="insight-grid reveal active">
            <div class="insight-card">
                <span class="insight-badge green">Staționară</span>
                <h2>White Noise</h2>
                <p>
                    Reprezintă un exemplu de referință pentru o serie staționară,
                    fără trend și fără sezonalitate. Este utilă pentru validarea
                    testelor de staționaritate și ca baseline teoretic.
                </p>
            </div>

            <div class="insight-card">
                <span class="insight-badge red">Nestaționară</span>
                <h2>Random Walk</h2>
                <p>
                    Este exemplul clasic de serie nestaționară. Chiar dacă variațiile
                    locale par mici, seria evoluează cumulativ și nu revine la o medie fixă.
                </p>
            </div>

            <div class="insight-card">
                <span class="insight-badge blue">Sezonieră</span>
                <h2>Temperatură demonstrativă</h2>
                <p>
                    Evidențiază un ciclu repetitiv anual și este folosită pentru a arăta
                    că metodologia nu se aplică doar la date satelitare, ci și la serii climatice.
                </p>
            </div>

            <div class="insight-card">
                <span class="insight-badge cyan">NDVI real</span>
                <h2>NDVI Agricol</h2>
                <p>
                    Prezintă sezonalitate puternică, corelată cu ciclurile vegetației.
                    Este unul dintre cele mai clare exemple reale de structură sezonieră din aplicație.
                </p>
            </div>

            <div class="insight-card">
                <span class="insight-badge gray">Fără sezonalitate clară</span>
                <h2>NDVI Urban</h2>
                <p>
                    Are valori mai scăzute și variații mai reduse. Prin comparație cu seria agricolă,
                    ilustrează un comportament mai slab structurat sezonier.
                </p>
            </div>

            <div class="insight-card">
                <span class="insight-badge teal">Vegetație urbană</span>
                <h2>NDVI Parc</h2>
                <p>
                    Ocupă o poziție intermediară între agricol și urban. Are nivel NDVI mai ridicat
                    decât zona urbană densă, dar o structură diferită față de terenul agricol.
                </p>
            </div>
        </section>

        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h2>Concluzii comparative</h2>

            <div class="method-box">
                <strong>White Noise vs Random Walk:</strong><br>
                Prima serie este staționară și servește drept referință teoretică, în timp ce
                Random Walk este nestaționar și ilustrează acumularea variațiilor în timp.
            </div>

            <div class="method-box">
                <strong>Temperatură vs NDVI Agricol:</strong><br>
                Ambele serii prezintă sezonalitate, dar în contexte diferite:
                temperatura reflectă ciclu climatic, iar NDVI agricol reflectă dinamica vegetației.
            </div>

            <div class="method-box">
                <strong>NDVI Urban vs NDVI Parc vs NDVI Agricol:</strong><br>
                Aceste trei serii oferă o comparație foarte relevantă între tipuri de acoperire a terenului:
                urban dens, vegetație urbană și teren agricol.
            </div>
        </section>

        {figure_card(
            fig,
            "Vizualizare comparativă integrată",
            "Graficul reunește seriile sintetice, climatice și NDVI pentru a evidenția diferențele structurale.",
            section_id="comparative_analysis_fig",
            yaxis_title="Valoare",
        )}
        """,
    )

@main_bp.route("/roi")
def roi_page():
    cards_html = ""
    for site_code, info in ROI_INFO.items():
        cards_html += f"""
        <div class="roi-card">
          <h2>{info["label"]}</h2>
          <div class="badge">{info["category"]}</div>
          <div class="roi-meta"><strong>Descriere:</strong><br>{info["description"]}</div>
          <div class="roi-meta"><strong>ROI / coordonate:</strong><br>{info["coords"]}</div>
          <div class="roi-meta"><strong>Comportament NDVI așteptat:</strong><br>{info["expected_ndvi"]}</div>
          <div class="roi-actions">
            <a class="btn-link" href="{info["route"]}">Vezi seria NDVI</a>
            <a class="btn-link secondary" href="/api/series?site={site_code}">Date JSON</a>
          </div>
        </div>
        """

    return render_template(
        "base.html",
        title="ROI – Regiuni de interes",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
          <h1>Regiuni de interes (ROI)</h1>
          <p class="muted">
            Această pagină descrie cele trei regiuni de interes utilizate în analiza NDVI:
            un spațiu verde urban, un teren agricol și o zonă urbană densă.
          </p>
          <div class="method-box">
            <strong>Rolul ROI-urilor:</strong><br>
            Fiecare ROI reprezintă un tip distinct de acoperire a terenului.
            Comparația dintre aceste zone permite observarea diferențelor de magnitudine,
            sezonalitate și anomalii.
          </div>
          <div class="roi-grid">{cards_html}</div>
        </section>
        """,
    )

@main_bp.route("/ml-features")
def ml_features_page():
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import plotly.express as px

        try:
            import umap
            umap_available = True
        except Exception:
            umap_available = False
    except Exception:
        return render_template(
            "base.html",
            title="ML Features",
            nav_html=render_nav(request.path),
            content="""
            <section class="card">
              <h1>ML Features</h1>
              <p class="muted">
                Pentru această pagină sunt necesare pachetele:
                <strong>scikit-learn</strong>, <strong>plotly</strong> și opțional <strong>umap-learn</strong>.
              </p>
            </section>
            """,
        )

    def build_windows(series: pd.Series, window_size: int = 24, step: int = 6):
        windows = []
        if len(series) < window_size:
            return windows

        for start in range(0, len(series) - window_size + 1, step):
            sub = series.iloc[start:start + window_size]
            windows.append({
                "series": sub,
                "start": sub.index.min().strftime("%Y-%m"),
                "end": sub.index.max().strftime("%Y-%m"),
            })
        return windows

    all_rows = []
    source_series = []

    df_white = generate_synthetic_series("white-noise")
    s_white = df_white.set_index("date")["value"].asfreq("MS")
    source_series.append(("White Noise", "Sintetică", s_white))

    df_rw = generate_synthetic_series("random-walk")
    s_rw = df_rw.set_index("date")["value"].asfreq("MS")
    source_series.append(("Random Walk", "Sintetică", s_rw))

    df_temp = generate_temperature_demo_series()
    s_temp = df_temp.set_index("date")["value"].asfreq("MS")
    source_series.append(("Temperatură demonstrativă", "Climatică", s_temp))

    df_ndvi = load_ndvi()
    for site, sub in df_ndvi.groupby("site"):
        s = prepare_monthly_series(sub)
        source_series.append((pretty_site_name(site), "NDVI", s))

    for series_name, category, series in source_series:
        windows = build_windows(series, window_size=24, step=6)

        for idx, w in enumerate(windows, start=1):
            features = extract_features(w["series"])
            label = classify_series_features(features)

            all_rows.append({
                "series_name": series_name,
                "category": category,
                "window_id": idx,
                "window_label": f"{series_name} [{w['start']} → {w['end']}]",
                "start": w["start"],
                "end": w["end"],
                "label": label,
                **features,
            })

    features_df = pd.DataFrame(all_rows)

    if features_df.empty:
        return render_template(
            "base.html",
            title="ML Features",
            nav_html=render_nav(request.path),
            content="""
            <section class="card">
              <h1>ML Features</h1>
              <p class="muted">Nu există suficiente date pentru a genera ferestre de analiză.</p>
            </section>
            """,
        )

    fig_features = px.scatter(
        features_df,
        x="mean",
        y="std",
        color="label",
        symbol="category",
        size="amplitude",
        hover_name="window_label",
        hover_data={
            "series_name": True,
            "category": True,
            "start": True,
            "end": True,
            "amplitude": ":.3f",
            "adf_pvalue": ":.4f",
            "anomalies": True,
            "label": True,
            "mean": False,
            "std": False,
            "window_label": False,
        },
        title="Feature Space: Mean vs Std",
    )
    fig_features.update_layout(
        xaxis_title="Mean",
        yaxis_title="Std",
        legend_title_text="Tip ML / Categorie",
    )

    X = features_df[["mean", "std", "amplitude", "anomalies"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(4, len(features_df))
    if n_clusters < 2:
        n_clusters = 1

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = features_df.copy()
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]
    pca_df["cluster"] = clusters.astype(str)

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        symbol="category",
        hover_name="window_label",
        hover_data={
            "series_name": True,
            "category": True,
            "start": True,
            "end": True,
            "label": True,
            "cluster": True,
            "PC1": False,
            "PC2": False,
            "window_label": False,
        },
        title="PCA + KMeans Clustering",
    )
    fig_pca.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title_text="Cluster / Categorie",
    )

    perplexity = max(2, min(10, len(features_df) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X_scaled)

    tsne_df = features_df.copy()
    tsne_df["TSNE1"] = X_tsne[:, 0]
    tsne_df["TSNE2"] = X_tsne[:, 1]

    fig_tsne = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color="label",
        symbol="category",
        hover_name="window_label",
        hover_data={
            "series_name": True,
            "category": True,
            "start": True,
            "end": True,
            "label": True,
            "TSNE1": False,
            "TSNE2": False,
            "window_label": False,
        },
        title="t-SNE Visualization",
    )
    fig_tsne.update_layout(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        legend_title_text="Tip ML / Categorie",
    )

    umap_html = ""
    if umap_available:
        n_neighbors = max(3, min(10, len(features_df) - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42,
        )
        X_umap = reducer.fit_transform(X_scaled)

        umap_df = features_df.copy()
        umap_df["UMAP1"] = X_umap[:, 0]
        umap_df["UMAP2"] = X_umap[:, 1]

        fig_umap = px.scatter(
            umap_df,
            x="UMAP1",
            y="UMAP2",
            color="label",
            symbol="category",
            hover_name="window_label",
            hover_data={
                "series_name": True,
                "category": True,
                "start": True,
                "end": True,
                "label": True,
                "UMAP1": False,
                "UMAP2": False,
                "window_label": False,
            },
            title="UMAP Visualization",
        )
        fig_umap.update_layout(
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            legend_title_text="Tip ML / Categorie",
        )

        umap_html = figure_card(
            fig_umap,
            "UMAP",
            "UMAP oferă o proiecție neliniară care păstrează atât structura locală, cât și o parte din structura globală a datelor.",
            section_id="ml_features_umap",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
        )
    else:
        umap_html = """
        <section class="card reveal active">
          <h1>UMAP</h1>
          <p class="muted">
            UMAP nu este disponibil în mediul curent. Instalează pachetul
            <code>umap-learn</code> pentru a activa această secțiune.
          </p>
        </section>
        """

    dtw_items = [(name, series) for name, _, series in source_series]
    dtw_df = pairwise_dtw_matrix(dtw_items, normalize=True)

    fig_dtw = px.imshow(
        dtw_df,
        text_auto=".2f",
        aspect="auto",
        title="DTW Distance Matrix",
    )
    fig_dtw.update_layout(
        xaxis_title="Serie",
        yaxis_title="Serie",
        coloraxis_colorbar_title="DTW",
    )

    preview_df = features_df[[
        "series_name", "category", "start", "end",
        "mean", "std", "amplitude", "adf_pvalue", "stationary", "anomalies", "label"
    ]].copy()

    preview_df = preview_df.sort_values(["series_name", "start"]).reset_index(drop=True)

    table_rows = ""
    for _, r in preview_df.iterrows():
        pvalue = "n/a" if pd.isna(r["adf_pvalue"]) else round(float(r["adf_pvalue"]), 4)

        badge_class = "blue"
        if r["label"] == "Stabilă":
            badge_class = "green"
        elif r["label"] == "Trending":
            badge_class = "red"
        elif r["label"] == "Mixtă":
            badge_class = "gray"

        table_rows += f"""
        <tr>
          <td>{r['series_name']}</td>
          <td>{r['category']}</td>
          <td>{r['start']}</td>
          <td>{r['end']}</td>
          <td>{round(float(r['mean']), 3)}</td>
          <td>{round(float(r['std']), 3)}</td>
          <td>{round(float(r['amplitude']), 3)}</td>
          <td>{pvalue}</td>
          <td>{r['stationary']}</td>
          <td>{int(r['anomalies'])}</td>
          <td><span class="badge {badge_class}">{r['label']}</span></td>
        </tr>
        """

    return render_template(
        "base.html",
        title="ML Features",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
          <h1>Feature Extraction pentru Machine Learning</h1>
          <p class="muted">
            În această pagină, seriile temporale sunt împărțite în
            <strong>ferestre glisante</strong>, iar fiecare fereastră este descrisă
            printr-un vector de features. Aceste reprezentări sunt apoi explorate
            prin metode de reducere a dimensionalității și măsuri de similaritate temporală.
          </p>
        </section>

        <section class="card reveal active">
          <h2>Reprezentare pe ferestre</h2>
          <p class="muted">
            Fiecare punct din graficele de mai jos reprezintă o fereastră de 24 luni
            extrasă dintr-o serie. Pentru fiecare fereastră se calculează:
            mean, std, amplitude, stationarity și anomaly count.
          </p>

          <div class="pipeline">
            <div class="pipeline-step"><span>24M</span><p>Window</p></div>
            <div class="pipeline-step"><span>μ</span><p>Mean</p></div>
            <div class="pipeline-step"><span>σ</span><p>Std</p></div>
            <div class="pipeline-step"><span>A</span><p>Amplitude</p></div>
            <div class="pipeline-step"><span>ADF</span><p>Stationarity</p></div>
            <div class="pipeline-step"><span>ML</span><p>Label</p></div>
          </div>
        </section>

        {figure_card(
            fig_features,
            "Feature Space",
            "Fiecare punct reprezintă o fereastră de serie în spațiul Mean–Std, cu amplitudinea redată prin dimensiunea markerului.",
            section_id="ml_features_scatter",
            xaxis_title="Mean",
            yaxis_title="Std",
        )}

        {figure_card(
            fig_pca,
            "PCA + KMeans",
            "Ferestrele de serie sunt proiectate în 2D prin PCA și grupate automat prin KMeans.",
            section_id="ml_features_pca",
            xaxis_title="PC1",
            yaxis_title="PC2",
        )}

        {figure_card(
            fig_tsne,
            "t-SNE",
            "t-SNE este o metodă neliniară de reducere a dimensionalității care păstrează vecinătățile locale dintre ferestrele de serie.",
            section_id="ml_features_tsne",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
        )}

        {umap_html}

        {figure_card(
            fig_dtw,
            "DTW Distance Matrix",
            "Matricea DTW compară seriile complete pe baza formei lor temporale, fiind robustă la deplasări și diferențe locale de aliniere.",
            section_id="ml_features_dtw",
            xaxis_title="Serie",
            yaxis_title="Serie",
        )}

        <section class="card reveal active">
          <h2>Feature Table</h2>
          <p class="muted">
            Tabelul de mai jos afișează features-urile extrase pentru fiecare fereastră temporală.
          </p>

          <div class="table-wrap">
            <table class="stats-table">
              <thead>
                <tr>
                  <th>Serie</th>
                  <th>Categorie</th>
                  <th>Start</th>
                  <th>End</th>
                  <th>Mean</th>
                  <th>Std</th>
                  <th>Amplitude</th>
                  <th>ADF p-value</th>
                  <th>Stationary</th>
                  <th>Anomalii</th>
                  <th>Tip ML</th>
                </tr>
              </thead>
              <tbody>
                {table_rows}
              </tbody>
            </table>
          </div>
        </section>

        <section class="card reveal active">
          <h2>Interpretare și utilizare în Machine Learning</h2>

          <div class="method-box">
            <strong>Tipuri de ferestre identificate:</strong><br>
            <span class="badge green">Stabilă</span> – fereastră staționară, cu variații reduse și fără trend semnificativ.<br>
            <span class="badge red">Trending</span> – fereastră nestaționară, cu trend sau variații mari în timp.<br>
            <span class="badge gray">Mixtă</span> – fereastră care combină stabilitate locală cu anomalii sau schimbări structurale.
          </div>

          <div class="method-box">
            <strong>Algoritmi utilizați:</strong><br>
            - ADF pentru evaluarea staționarității<br>
            - STL pentru detectarea anomaliilor<br>
            - PCA, t-SNE și UMAP pentru reducerea dimensionalității<br>
            - KMeans pentru gruparea automată a ferestrelor<br>
            - DTW pentru compararea directă a seriilor pe baza formei temporale
          </div>

          <div class="method-box">
            <strong>Rol practic:</strong><br>
            Aceste metode permit explorarea similarității dintre segmente de serie,
            identificarea clusterelor și compararea tipurilor de vegetație dincolo de simpla inspecție vizuală.
          </div>
        </section>
        """,
    )

@main_bp.route("/spectral-indices")
def spectral_indices_page():
    available_indices = list_indices()

    selected_index = request.args.get("index", "NDVI")
    if selected_index not in available_indices:
        selected_index = available_indices[0] if available_indices else "NDVI"

    try:
        df = load_index_dataframe(selected_index)
    except Exception as exc:
        return render_template(
            "base.html",
            title="Indici spectrali",
            nav_html=render_nav(request.path),
            content=f"""
            <section class="card reveal active">
              <h1>Indici spectrali</h1>
              <p class="muted">Eroare la încărcarea datelor: {exc}</p>
            </section>
            """,
        )

    options_html = ""
    for index_name in available_indices:
        selected = "selected" if index_name == selected_index else ""
        options_html += f"<option value='{index_name}' {selected}>{index_name}</option>"

    description = INDEX_DESCRIPTIONS.get(selected_index, "Indice spectral utilizat în analiza vegetației.")

    fig = px.line(
        df,
        x="date",
        y="value",
        color="roi",
        markers=True,
        title=f"Serie temporală {selected_index} – ROI1 vs ROI2",
    )

    stats = df.groupby("roi")["value"].agg(["mean", "std", "min", "max"]).reset_index()

    table_rows = ""
    for _, row in stats.iterrows():
        table_rows += f"""
        <tr>
          <td>{row['roi']}</td>
          <td>{round(float(row['mean']), 4)}</td>
          <td>{round(float(row['std']), 4)}</td>
          <td>{round(float(row['min']), 4)}</td>
          <td>{round(float(row['max']), 4)}</td>
        </tr>
        """

    return render_template(
        "base.html",
        title="Indici spectrali",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
          <h1>Analiza indicilor spectrali</h1>
          <p class="muted">
            Această pagină extinde analiza de la NDVI către mai mulți indici spectrali
            asociați vegetației, precum EVI, SAVI, GNDVI, GCI, MSI și AVI.
          </p>

          <form method="get" class="method-box">
            <label for="index"><strong>Selectează indicele spectral:</strong></label><br><br>
            <select name="index" id="index" onchange="this.form.submit()" class="select-input">
              {options_html}
            </select>
          </form>

          <div class="method-box">
            <strong>{selected_index}:</strong><br>
            {description}
          </div>
        </section>

        {figure_card(
            fig,
            f"{selected_index} – serie temporală",
            "Seria este obținută prin media valorilor pixelilor din fiecare moment temporal al stack-ului .npy.",
            section_id="spectral_index_fig",
            xaxis_title="Data",
            yaxis_title=selected_index,
        )}

        <section class="card reveal active">
          <h2>Statistici descriptive</h2>
          <div class="table-wrap">
            <table class="stats-table">
              <thead>
                <tr>
                  <th>ROI</th>
                  <th>Mean</th>
                  <th>Std</th>
                  <th>Min</th>
                  <th>Max</th>
                </tr>
              </thead>
              <tbody>
                {table_rows}
              </tbody>
            </table>
          </div>
        </section>
        """,
    )

@main_bp.route("/cross-index-analysis")
def cross_index_analysis_page():
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import plotly.express as px
    except Exception:
        return render_template(
            "base.html",
            title="Cross-Index Analysis",
            nav_html=render_nav(request.path),
            content="""
            <section class="card reveal active">
              <h1>Cross-Index Analysis</h1>
              <p class="muted">
                Pentru această pagină sunt necesare pachetele scikit-learn și plotly.
              </p>
            </section>
            """,
        )

    selected_roi = request.args.get("roi", "roi1")
    if selected_roi not in ["roi1", "roi2"]:
        selected_roi = "roi1"

    wide_df = build_indices_wide_dataframe(roi=selected_roi)

    if wide_df.empty or wide_df.shape[1] < 2:
        return render_template(
            "base.html",
            title="Cross-Index Analysis",
            nav_html=render_nav(request.path),
            content="""
            <section class="card reveal active">
              <h1>Cross-Index Analysis</h1>
              <p class="muted">
                Nu există suficiente date pentru analiza comparativă între indici.
              </p>
            </section>
            """,
        )

    roi_options = ""
    for roi in ["roi1", "roi2"]:
        selected = "selected" if roi == selected_roi else ""
        roi_options += f"<option value='{roi}' {selected}>{roi.upper()}</option>"

    corr = wide_df.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title=f"Matrice de corelație între indici – {selected_roi.upper()}",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig_corr.update_layout(
        xaxis_title="Indice spectral",
        yaxis_title="Indice spectral",
        coloraxis_colorbar_title="Corelație",
    )

    X = wide_df.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    X = X.dropna(axis=0, how="any")
    X = X.clip(lower=-1e6, upper=1e6)

    if X.empty or X.shape[1] < 2:
       return render_template(
        "base.html",
        title="Cross-Index Analysis",
        nav_html=render_nav(request.path),
        content="""
        <section class="card reveal active">
          <h1>Cross-Index Analysis</h1>
          <p class="muted">
            Datele nu sunt suficiente sau conțin prea multe valori invalide pentru PCA.
          </p>
        </section>
        """,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "date": X.index.strftime("%Y-%m"),
    })

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        hover_name="date",
        title=f"PCA pe indici spectrali – {selected_roi.upper()}",
    )
    fig_pca.update_layout(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% varianță)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% varianță)",
    )

    pairs = []
    indices = corr.columns.tolist()

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            pairs.append({
                "index_a": indices[i],
                "index_b": indices[j],
                "corr": float(corr.loc[indices[i], indices[j]]),
            })

    pairs_df = pd.DataFrame(pairs)
    pairs_df["abs_corr"] = pairs_df["corr"].abs()
    pairs_df = pairs_df.sort_values("abs_corr", ascending=False)

    table_rows = ""
    for _, row in pairs_df.iterrows():
        strength = "Ridicată" if row["abs_corr"] >= 0.8 else "Medie" if row["abs_corr"] >= 0.5 else "Scăzută"

        table_rows += f"""
        <tr>
          <td>{row['index_a']}</td>
          <td>{row['index_b']}</td>
          <td>{round(row['corr'], 4)}</td>
          <td>{strength}</td>
        </tr>
        """

    strongest = pairs_df.iloc[0]
    weakest = pairs_df.iloc[-1]

    return render_template(
        "base.html",
        title="Cross-Index Analysis",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
          <h1>Cross-Index Comparative Analysis</h1>
          <p class="muted">
            Această pagină compară indicii spectrali între ei pentru același ROI.
            Scopul este identificarea indicilor care au comportament temporal similar
            și a celor care surprind informații diferite despre vegetație.
          </p>

          <form method="get" class="method-box">
            <label for="roi"><strong>Selectează ROI:</strong></label><br><br>
            <select name="roi" id="roi" onchange="this.form.submit()" class="select-input">
              {roi_options}
            </select>
          </form>
        </section>

        {figure_card(
            fig_corr,
            "Matrice de corelație între indici",
            "Valorile apropiate de 1 indică evoluții similare între indici, iar valorile mai mici indică diferențe de comportament temporal.",
            section_id="cross_index_corr",
            xaxis_title="Indice spectral",
            yaxis_title="Indice spectral",
        )}

        {figure_card(
            fig_pca,
            "PCA pe indici spectrali",
            "Fiecare punct reprezintă un moment temporal, descris prin valorile tuturor indicilor spectrali disponibili.",
            section_id="cross_index_pca",
            xaxis_title="PC1",
            yaxis_title="PC2",
        )}

        <section class="card reveal active">
          <h2>Tabel similaritate între indici</h2>
          <div class="table-wrap">
            <table class="stats-table">
              <thead>
                <tr>
                  <th>Indice A</th>
                  <th>Indice B</th>
                  <th>Corelație</th>
                  <th>Similaritate</th>
                </tr>
              </thead>
              <tbody>
                {table_rows}
              </tbody>
            </table>
          </div>
        </section>

        <section class="card reveal active">
          <h2>Interpretare automată</h2>

          <div class="method-box">
            <strong>Cea mai mare similaritate:</strong><br>
            Perechea <strong>{strongest['index_a']} – {strongest['index_b']}</strong>
            are corelația <strong>{round(float(strongest['corr']), 4)}</strong>,
            ceea ce indică un comportament temporal foarte apropiat.
          </div>

          <div class="method-box">
            <strong>Cea mai mică similaritate:</strong><br>
            Perechea <strong>{weakest['index_a']} – {weakest['index_b']}</strong>
            are corelația <strong>{round(float(weakest['corr']), 4)}</strong>,
            sugerând că acești indici surprind aspecte diferite ale vegetației.
          </div>

          <div class="method-box">
            <strong>Rol metodologic:</strong><br>
            Analiza cross-index ajută la identificarea indicilor redundanți și a indicilor
            complementari. Astfel, aplicația nu se limitează la NDVI, ci analizează
            comportamentul multispectral al vegetației.
          </div>
        </section>
        """,
    )

@main_bp.route("/methodology")
def methodology_page():
    df = load_ndvi()
    date_min = df["date"].min().strftime("%Y-%m-%d")
    date_max = df["date"].max().strftime("%Y-%m-%d")

    return render_template(
        "base.html",
        title="Metodologie",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
            <h1>Cadrul metodologic</h1>
            <p class="muted">
                Aplicația este concepută ca o platformă de analiză și vizualizare a seriilor temporale,
                folosind atât date reale (NDVI Sentinel-2), cât și serii sintetice și climatice.
            </p>
        </section>

        <div class="method-grid">

            <div class="method-card">
                <div class="badge green">Staționare</div>
                <h2>Serii staționare</h2>
                <p>
                    Serii fără trend și fără sezonalitate, cu medie și varianță aproximativ constante.
                    Sunt folosite ca referință teoretică pentru validarea analizelor.
                </p>
                <a href="/synthetic/white-noise" class="btn-link">Exemplu: White Noise</a>
            </div>

            <div class="method-card">
                <div class="badge red">Nestaționare</div>
                <h2>Serii nestaționare</h2>
                <p>
                    Serii care prezintă trend sau schimbări structurale în timp și necesită atenție
                    specială la analiză și forecast.
                </p>
                <a href="/synthetic/random-walk" class="btn-link">Exemplu: Random Walk</a>
            </div>

            <div class="method-card">
                <div class="badge blue">Sezonalitate</div>
                <h2>Serii sezoniere</h2>
                <p>
                    Serii cu pattern repetitiv, lunar sau anual. În aplicație, acest tip este ilustrat
                    prin temperatură și NDVI agricol.
                </p>
                <a href="/agricol" class="btn-link">Exemplu: NDVI agricol</a>
            </div>

            <div class="method-card">
                <div class="badge gray">Fără sezonalitate clară</div>
                <h2>Serii fără sezonalitate</h2>
                <p>
                    Serii care nu prezintă ciclu periodic evident și la care comportamentul este dominat
                    de variații locale sau structură slabă.
                </p>
                <a href="/urban" class="btn-link">Exemplu: NDVI urban</a>
            </div>

        </div>

        <section class="card reveal active">
            <h2>Pipeline analitic</h2>

            <div class="pipeline">
                <div class="pipeline-step">
                    <span>1</span>
                    <p>Încărcarea sau generarea seriei</p>
                </div>

                <div class="pipeline-step">
                    <span>2</span>
                    <p>Analiză descriptivă</p>
                </div>

                <div class="pipeline-step">
                    <span>3</span>
                    <p>Test ADF pentru staționaritate</p>
                </div>

                <div class="pipeline-step">
                    <span>4</span>
                    <p>Descompunere STL</p>
                </div>

                <div class="pipeline-step">
                    <span>5</span>
                    <p>Detectare anomalii</p>
                </div>

                <div class="pipeline-step">
                    <span>6</span>
                    <p>Forecast ARIMA / LSTM</p>
                </div>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Date utilizate</h2>
            <div class="method-box">
                <strong>Perioada analizată pentru NDVI:</strong><br>
                <strong>{date_min}</strong> – <strong>{date_max}</strong>
            </div>

            <div class="method-box">
                <strong>Tipuri de serii incluse:</strong><br>
                1. Serii sintetice: white noise, random walk, trend liniar, sezonalitate, trend + sezonalitate.<br>
                2. Serii climatice demonstrative: temperatură lunară.<br>
                3. Serii NDVI Sentinel-2: parc urban, teren agricol, zonă urbană densă.
            </div>

            <div class="method-box">
                <strong>Justificarea seriilor sintetice:</strong><br>
                Seriile sintetice sunt utilizate pentru a controla proprietățile statistice ale datelor și
                pentru a valida metodele de analiză în cazuri teoretice cunoscute. De exemplu, seria de tip
                White Noise este inclusă ca exemplu de referință pentru o serie staționară, iar Random Walk
                pentru o serie nestaționară.
            </div>
        </section>

        <section class="card reveal active">
            <h2>Poziționarea aplicației</h2>
            <p class="muted">
                NDVI rămâne studiul principal de caz, însă aplicația demonstrează că aceleași tehnici
                pot fi aplicate și altor categorii de serii temporale, ceea ce crește valoarea metodologică a proiectului.
            </p>
        </section>
        """,
    )