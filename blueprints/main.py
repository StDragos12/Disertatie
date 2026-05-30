from statistics import variance
import pandas as pd
from flask import Blueprint, render_template, request, jsonify
from tslearn.metrics import dtw
from sklearn.ensemble import IsolationForest
import numpy as np
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import plotly.express as px
from services.indices_service import (
    list_indices,
    load_index_dataframe,
    build_indices_wide_dataframe,
    INDEX_DESCRIPTIONS,
    smooth_series,
)
from services.precomputed_ml_service import (
    DEFAULT_PIXEL_COUNTS,
    load_precomputed_ml_payload,
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
    pairwise_amss_matrix,
)

main_bp = Blueprint("main", __name__)




@main_bp.route("/")
def home():
    return render_template(
        "base.html",
        title="NDVI Viz",
        nav_html=render_nav(request.path),
        content="""
        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h1>Platformă pentru analiza spațio-temporală a vegetației</h1>
            <p class="muted">
                Aplicația combină indici spectrali, analiză de serii temporale, machine learning pe pixeli
                și forecast pentru interpretarea evoluției vegetației în regiunile analizate.
            </p>

            <div class="method-box">
                <strong>Ce poți analiza:</strong><br>
                evoluția indicilor spectrali, componentele temporale, anomaliile, gruparea pixelilor
                în clustere și estimarea evoluției viitoare prin SARIMA și LSTM.
            </div>

            <div class="roi-actions">
                <a class="btn-link" href="/series-catalog">Catalog module</a>
                <a class="btn-link secondary" href="/spectral-indices">Indici spectrali</a>
                <a class="btn-link secondary" href="/ml-features">ML pe pixeli</a>
                <a class="btn-link secondary" href="/forecast-arima">Forecast</a>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Module principale</h2>

            <div class="catalog-grid">

                <div class="catalog-card">
                    <div class="badge green">Indici</div>
                    <h2>Analiză spectrală</h2>
                    <p class="muted">
                        Vizualizează și compară indicii NDVI, NDMI, SAVI, AVI, EVI și GNDVI.
                    </p>
                    <ul>
                        <li><a href="/spectral-indices">Deschide modulul</a></li>
                    </ul>
                </div>

                <div class="catalog-card highlight-card">
                    <div class="badge red">ML</div>
                    <h2>Hărți de risc</h2>
                    <p class="muted">
                        Identifică zone cu comportament temporal similar sau atipic la nivel de pixel.
                    </p>
                    <ul>
                        <li><a href="/ml-features">Deschide modulul</a></li>
                    </ul>
                </div>

                <div class="catalog-card">
                    <div class="badge purple">Forecast</div>
                    <h2>Predicție</h2>
                    <p class="muted">
                        Estimează evoluția viitoare prin ARIMA/SARIMA și LSTM.
                    </p>
                    <ul>
                        <li><a href="/forecast-arima">Forecast ARIMA / SARIMA</a></li>
                        <li><a href="/forecast-lstm">Forecast LSTM</a></li>
                    </ul>
                </div>

                <div class="catalog-card">
                    <div class="badge gray">Metodologie</div>
                    <h2>Explicații</h2>
                    <p class="muted">
                        Prezintă pipeline-ul analitic și rolul fiecărui modul în aplicație.
                    </p>
                    <ul>
                        <li><a href="/methodology">Deschide metodologia</a></li>
                    </ul>
                </div>

            </div>
        </section>
        """,
    )




@main_bp.route("/series-catalog")
def series_catalog():
    return render_template(
        "base.html",
        title="Catalog module",
        nav_html=render_nav(request.path),
        content="""
        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h1>Catalogul modulelor aplicației</h1>
            <p class="muted">
                Această pagină oferă o vedere de ansamblu asupra modulelor principale ale platformei:
                date satelitare, indici spectrali, serii sintetice, analiză temporală, machine learning,
                forecast și metodologie. Fiecare card duce către pagina principală a modulului, fără a încărca
                inutil catalogul cu toate combinațiile de indici și ROI-uri.
            </p>

            <div class="method-box">
                <strong>Recomandare de utilizare:</strong><br>
                Pornește de la indicii spectrali, verifică analiza temporală, apoi folosește modulul ML
                pentru hărți de risc și modulul de forecast pentru estimarea evoluției viitoare.
            </div>
        </section>

        <section class="card reveal active">
            <h2>Fluxul principal</h2>

            <div class="pipeline">
                <div class="pipeline-step">
                    <span>1</span>
                    <p>Date și ROI-uri</p>
                </div>

                <div class="pipeline-step">
                    <span>2</span>
                    <p>Indici spectrali</p>
                </div>

                <div class="pipeline-step">
                    <span>3</span>
                    <p>Analiză temporală</p>
                </div>

                <div class="pipeline-step">
                    <span>4</span>
                    <p>ML pe pixeli</p>
                </div>

                <div class="pipeline-step">
                    <span>5</span>
                    <p>Forecast și interpretare</p>
                </div>
            </div>
        </section>

        <div class="catalog-grid">

            <div class="catalog-card">
                <div class="badge blue">Date</div>
                <h2>Date și ROI-uri</h2>
                <p class="muted">
                    Descrierea regiunilor analizate și accesul către datele agregate folosite de aplicație.
                </p>
                <ul>
                    <li><a href="/roi">Deschide modulul ROI</a></li>
                    <li><a href="/api/series?index=NDVI&roi=roi1">Exemplu date JSON</a></li>
                </ul>
            </div>

            <div class="catalog-card">
                <div class="badge green">Indici</div>
                <h2>Indici spectrali</h2>
                <p class="muted">
                    Vizualizarea și compararea indicilor NDVI, NDMI, SAVI, AVI, EVI și GNDVI.
                </p>
                <ul>
                    <li><a href="/spectral-indices">Deschide analiza indicilor</a></li>
                    <li><a href="/cross-index-analysis">Analiză cross-index</a></li>
                </ul>
            </div>

            <div class="catalog-card">
                <div class="badge cyan">Serii</div>
                <h2>Serii sintetice</h2>
                <p class="muted">
                    Exemple controlate pentru explicarea staționarității, trendului, sezonalității și zgomotului.
                </p>
                <ul>
                    <li><a href="/synthetic">Catalog serii sintetice</a></li>
                    <li><a href="/temperature-demo">Temperatură demonstrativă</a></li>
                </ul>
            </div>

            <div class="catalog-card">
                <div class="badge teal">Temporal</div>
                <h2>Analiză temporală</h2>
                <p class="muted">
                    Module pentru staționaritate, descompunere STL, trend, sezonalitate și anomalii.
                </p>
                <ul>
                    <li><a href="/stationarity">Staționaritate</a></li>
                    <li><a href="/decompose">Descompunere STL</a></li>
                    <li><a href="/trend">Trend</a></li>
                    <li><a href="/seasonality">Sezonalitate</a></li>
                    <li><a href="/anomalies">Anomalii</a></li>
                </ul>
            </div>

            <div class="catalog-card highlight-card">
                <div class="badge red">ML</div>
                <h2>Machine Learning pe pixeli</h2>
                <p class="muted">
                    Clustering, detecție de anomalii, PCA, t-SNE, UMAP, DTW, profil temporal pe cluster
                    și hărți de risc/anomalie temporală.
                </p>
                <ul>
                    <li><a href="/ml-features">Deschide modulul ML</a></li>
                </ul>
            </div>

            <div class="catalog-card">
                <div class="badge purple">Forecast</div>
                <h2>Prognoză temporală</h2>
                <p class="muted">
                    Estimarea evoluției viitoare prin modele ARIMA/SARIMA și LSTM.
                </p>
                <ul>
                    <li><a href="/forecast-arima">Forecast ARIMA / SARIMA</a></li>
                    <li><a href="/forecast-lstm">Forecast LSTM</a></li>
                </ul>
            </div>

            <div class="catalog-card">
                <div class="badge gray">Metodologie</div>
                <h2>Metodologie</h2>
                <p class="muted">
                    Explicarea pipeline-ului, a ipotezelor, a metodelor folosite și a rolului fiecărui modul.
                </p>
                <ul>
                    <li><a href="/methodology">Deschide metodologia</a></li>
                    <li><a href="/debug">Debug date</a></li>
                </ul>
            </div>

        </div>

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
    available_indices = [
        "NDVI",
        "NDMI",
        "SAVI",
        "AVI",
        "EVI",
        "GNDVI",
    ]

    selected_index = request.args.get("index", "NDVI").upper()

    if selected_index not in available_indices:
        selected_index = "NDVI"

    roi = request.args.get("roi", "roi1").lower()

    if roi not in ["roi1", "roi2"]:
        roi = "roi1"

    try:
        pixel_count = int(request.args.get("pixels", "1000"))
    except Exception:
        pixel_count = 1000

    if pixel_count not in DEFAULT_PIXEL_COUNTS:
        pixel_count = 1000

    index_options = ""

    for index_name in available_indices:
        selected = "selected" if selected_index == index_name else ""

        index_options += f"""
        <option value="{index_name}" {selected}>{index_name}</option>
        """

    roi_options = ""

    for roi_name in ["roi1", "roi2"]:
        selected = "selected" if roi == roi_name else ""

        roi_options += f"""
        <option value="{roi_name}" {selected}>{roi_name.upper()}</option>
        """

    pixel_options = ""

    for option in DEFAULT_PIXEL_COUNTS:
        selected = "selected" if pixel_count == option else ""

        pixel_options += f"""
        <option value="{option}" {selected}>{option} pixeli</option>
        """

    try:
        payload = load_precomputed_ml_payload(
            selected_index,
            roi,
            pixel_count,
        )

    except Exception as exc:
        return render_template(
            "base.html",
            title="ML pe pixeli",
            nav_html=render_nav(request.path),
            content=f"""
            <section class="card reveal active">
                <div class="card-top-line"></div>
                <h1>Analiză ML pe pixeli</h1>

                <p class="muted">
                    Rezultatele precompute nu sunt disponibile pentru combinația selectată.
                </p>

                <form method="get" class="method-box">
                    <label><strong>Indice spectral:</strong></label><br>
                    <select name="index" onchange="this.form.submit()" class="select-input">
                        {index_options}
                    </select>

                    <br><br>

                    <label><strong>ROI:</strong></label><br>
                    <select name="roi" onchange="this.form.submit()" class="select-input">
                        {roi_options}
                    </select>

                    <br><br>

                    <label><strong>Număr pixeli:</strong></label><br>
                    <select name="pixels" onchange="this.form.submit()" class="select-input">
                        {pixel_options}
                    </select>
                </form>

                <div class="method-box">
                    <strong>Detalii:</strong><br>
                    {exc}
                </div>
            </section>
            """,
        )

    metadata = payload["metadata"]
    metrics = payload["metrics"]
    highlights = payload["highlights"]

    fig_cluster_map = px.imshow(
        payload["cluster_grid"],
        color_continuous_scale="Turbo",
        aspect="equal",
        title="Hartă clustere pixeli",
        zmin=1,
        zmax=metadata["n_clusters"],
    )

    fig_cluster_map.update_layout(
        height=650,
        xaxis_title="Coloană pixel",
        yaxis_title="Linie pixel",
        coloraxis_colorbar_title="Cluster",
    )

    fig_risk_map = px.imshow(
        payload["risk_grid"],
        color_continuous_scale="Turbo",
        aspect="equal",
        title="Hartă risc/anomalie temporală",
    )

    fig_risk_map.update_layout(
        height=650,
        xaxis_title="Coloană pixel",
        yaxis_title="Linie pixel",
        coloraxis_colorbar_title="Scor anomalie",
    )

    cluster_profile_fig = go.Figure()

    for profile in payload["cluster_profiles"]:
        cluster_profile_fig.add_trace(
            go.Scatter(
                x=profile["dates"],
                y=profile["values"],
                mode="lines",
                name=profile["label"],
            )
        )

    cluster_profile_fig.update_layout(
        title=f"Profil temporal mediu pe cluster – {selected_index}",
        xaxis_title="Data",
        yaxis_title=selected_index,
        height=560,
        hovermode="x unified",
        legend_title_text="Cluster",
    )

    pca_df = pd.DataFrame(payload["pca_points"])

    fig_pca = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="cluster_label",
        hover_data={
            "pixel_id": True,
            "risk_score": ":.4f",
            "mean": ":.4f",
            "amplitude": ":.4f",
            "window_start": True,
            "window_end": True,
        },
        title="PCA 3D pe semnături temporale ale pixelilor",
    )

    fig_pca.update_layout(
        height=650,
        legend_title_text="Cluster",
    )

    tsne_html = ""

    if payload.get("tsne_points"):
        tsne_df = pd.DataFrame(payload["tsne_points"])

        fig_tsne = px.scatter(
            tsne_df,
            x="TSNE1",
            y="TSNE2",
            color="cluster_label",
            hover_data={
                "pixel_id": True,
                "risk_score": ":.4f",
                "mean": ":.4f",
                "amplitude": ":.4f",
            },
            title="t-SNE pe semnături temporale",
        )

        fig_tsne.update_layout(
            height=600,
            legend_title_text="Cluster",
        )

        tsne_html = figure_card(
            fig_tsne,
            "t-SNE pe semnături temporale",
            "Vizualizare neliniară a similarității dintre ferestrele temporale extrase din pixeli.",
            section_id="tsne_pixels",
        )

    umap_html = ""

    if payload.get("umap_points"):
        umap_df = pd.DataFrame(payload["umap_points"])

        fig_umap = px.scatter(
            umap_df,
            x="UMAP1",
            y="UMAP2",
            color="cluster_label",
            hover_data={
                "pixel_id": True,
                "risk_score": ":.4f",
                "mean": ":.4f",
                "amplitude": ":.4f",
            },
            title="UMAP pe semnături temporale",
        )

        fig_umap.update_layout(
            height=600,
            legend_title_text="Cluster",
        )

        umap_html = figure_card(
            fig_umap,
            "UMAP",
            "Reducere dimensională UMAP aplicată semnăturilor temporale ale pixelilor.",
            section_id="ml_umap",
        )

    dtw_html = ""

    if payload.get("dtw", {}).get("matrix"):
        fig_dtw = px.imshow(
            payload["dtw"]["matrix"],
            x=payload["dtw"]["labels"],
            y=payload["dtw"]["labels"],
            text_auto=".2f",
            color_continuous_scale="Viridis",
            title=f"DTW între profilele medii ale clusterelor – {selected_index}",
        )

        fig_dtw.update_layout(
            height=560,
            xaxis_title="Cluster",
            yaxis_title="Cluster",
            coloraxis_colorbar_title="DTW",
        )

        dtw_html = figure_card(
            fig_dtw,
            "DTW între clustere",
            "Valorile mici indică profile temporale asemănătoare, iar valorile mari indică diferențe mai pronunțate.",
            section_id="dtw_clusters",
            xaxis_title="Cluster",
            yaxis_title="Cluster",
        )

    amss_html = ""

    try:
        amss_rows = []

        for profile in payload.get("cluster_profiles", []):
            label = profile["label"]
            values = pd.Series(profile["values"]).astype(float)

            amss_rows.append(
                (
                    label,
                    values,
                )
            )

        if len(amss_rows) >= 2:
            amss_matrix = pairwise_amss_matrix(
                amss_rows
            )

            amss_labels = [
                item[0]
                for item in amss_rows
            ]

            fig_amss = px.imshow(
                amss_matrix,
                x=amss_labels,
                y=amss_labels,
                text_auto=".2f",
                color_continuous_scale="Cividis",
                title=f"AMSS între profilele medii ale clusterelor – {selected_index}",
            )

            fig_amss.update_layout(
                height=560,
                xaxis_title="Cluster",
                yaxis_title="Cluster",
                coloraxis_colorbar_title="AMSS",
            )

            amss_html = figure_card(
                fig_amss,
                "AMSS între clustere",
                (
                    "Matricea AMSS compară forma generală a profilelor temporale medii. "
                    "Valorile mici indică evoluții asemănătoare, iar valorile mari indică "
                    "diferențe mai pronunțate între clustere."
                ),
                section_id="amss_clusters",
                xaxis_title="Cluster",
                yaxis_title="Cluster",
            )

    except Exception as exc:
        amss_html = f"""
        <section class="card reveal active">
            <h2>AMSS între clustere</h2>
            <p class="muted">
                Matricea AMSS nu a putut fi calculată pentru această selecție.
            </p>
            <div class="method-box">
                <strong>Detalii:</strong><br>
                {exc}
            </div>
        </section>
        """

    pixel_compare = payload["pixel_compare"]

    fig_pixel_compare = go.Figure()

    fig_pixel_compare.add_trace(
        go.Scatter(
            x=pixel_compare["risk"]["dates"],
            y=pixel_compare["risk"]["values"],
            mode="lines",
            name="Pixel de urmărit",
        )
    )

    fig_pixel_compare.add_trace(
        go.Scatter(
            x=pixel_compare["representative"]["dates"],
            y=pixel_compare["representative"]["values"],
            mode="lines",
            name="Pixel reprezentativ",
        )
    )

    fig_pixel_compare.update_layout(
        title=f"Pixel de urmărit vs pixel reprezentativ (DTW={pixel_compare['dtw']})",
        xaxis_title="Data",
        yaxis_title=selected_index,
        height=540,
        showlegend=True,
        hovermode="x unified",
    )

    cluster_rows = ""

    for row in payload["cluster_summary"]:
        cluster_rows += f"""
        <tr>
            <td>Cluster {row["cluster"]}</td>
            <td>{row["sample_pixels"]}</td>
            <td>{row["mapped_pixels"]}</td>
            <td>{row["windows"]}</td>
            <td>{row["mean"]}</td>
            <td>{row["amplitude"]}</td>
            <td>{row["trend"]}</td>
            <td>{row["risk_score"]}</td>
            <td>{row["interpretation"]}</td>
        </tr>
        """

    return render_template(
        "base.html",
        title="ML pe pixeli",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h1>Analiză ML pe pixeli și hartă de risc</h1>

            <p class="muted">
                Rezultatele sunt preprocesate prin Cloud Run Job și încărcate din Cloud Storage.
                Pagina afișează rapid hărțile și graficele fără recalcularea live a modelului.
            </p>

            <form method="get" class="method-box">
                <label><strong>Indice spectral:</strong></label><br>
                <select name="index" onchange="this.form.submit()" class="select-input">
                    {index_options}
                </select>

                <br><br>

                <label><strong>ROI:</strong></label><br>
                <select name="roi" onchange="this.form.submit()" class="select-input">
                    {roi_options}
                </select>

                <br><br>

                <label><strong>Număr pixeli pentru antrenare:</strong></label><br>
                <select name="pixels" onchange="this.form.submit()" class="select-input">
                    {pixel_options}
                </select>
            </form>

            <div class="method-box">
                <strong>Configurație curentă:</strong><br>
                Indice: <strong>{selected_index}</strong><br>
                ROI: <strong>{roi.upper()}</strong><br>
                Eșantion model: <strong>{metadata["pixel_count"]}</strong> pixeli<br>
                Pixeli validați mapați: <strong>{metadata["mapped_pixels"]}</strong><br>
                Ferestre temporale: <strong>{metadata["windows_extracted"]}</strong>
            </div>
        </section>

                <section class="card reveal active">
            <div class="section-heading">
                <span class="section-kicker">Interpretare automată</span>
                <h2>Rezumat operațional al analizei</h2>
                <p class="muted">
                    Sinteză a principalelor rezultate obținute pentru indicele
                    <strong>{selected_index}</strong>, regiunea <strong>{roi.upper()}</strong>
                    și eșantionul de <strong>{metadata["pixel_count"]}</strong> pixeli.
                </p>
            </div>

            <div class="summary-dashboard">
                <div class="summary-tile summary-tile-main">
                    <div class="summary-tile-top">
                        <span class="summary-icon">01</span>
                        <span class="summary-label">Cluster dominant</span>
                    </div>
                    <h3>Cluster {highlights["dominant_cluster"]}</h3>
                    <p>
                        Cea mai extinsă clasă temporală din hartă, cu
                        <strong>{highlights["dominant_pixels"]}</strong> pixeli mapați.
                    </p>
                </div>

                <div class="summary-tile">
                    <div class="summary-tile-top">
                        <span class="summary-icon">02</span>
                        <span class="summary-label warning">Zonă de urmărit</span>
                    </div>
                    <h3>Cluster {highlights["risk_cluster"]}</h3>
                    <p>
                        Clusterul cu cel mai ridicat scor mediu de anomalie:
                        <strong>{highlights["risk_score"]}</strong>.
                    </p>
                </div>

                <div class="summary-tile">
                    <div class="summary-tile-top">
                        <span class="summary-icon">03</span>
                        <span class="summary-label seasonal">Variabilitate sezonieră</span>
                    </div>
                    <h3>Cluster {highlights["seasonal_cluster"]}</h3>
                    <p>
                        Clusterul cu cea mai mare amplitudine medie:
                        <strong>{highlights["seasonal_amplitude"]}</strong>.
                    </p>
                </div>

                <div class="summary-tile">
                    <div class="summary-tile-top">
                        <span class="summary-icon">04</span>
                        <span class="summary-label muted-label">Nivel mediu redus</span>
                    </div>
                    <h3>Cluster {highlights["low_vegetation_cluster"]}</h3>
                    <p>
                        Clusterul cu cea mai mică valoare medie a indicelui:
                        <strong>{highlights["low_vegetation_mean"]}</strong>.
                    </p>
                </div>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Rezumat tehnic ML</h2>

            <div class="table-wrap">
                <table class="stats-table">
                    <tbody>
                        <tr>
                            <td>Silhouette Score</td>
                            <td>{metrics["silhouette"]}</td>
                        </tr>
                        <tr>
                            <td>Calinski-Harabasz</td>
                            <td>{metrics["calinski_harabasz"]}</td>
                        </tr>
                        <tr>
                            <td>Davies-Bouldin</td>
                            <td>{metrics["davies_bouldin"]}</td>
                        </tr>
                        <tr>
                            <td>Model precompute</td>
                            <td>Cloud Run Job + Cloud Storage</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Profilul clusterelor</h2>

            <p class="muted">
                Interpretarea este automată și se bazează pe medie, amplitudine, trend și scor de anomalie.
            </p>

            <div class="table-wrap">
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Cluster</th>
                            <th>Pixeli eșantion</th>
                            <th>Pixeli mapați</th>
                            <th>Ferestre</th>
                            <th>Medie</th>
                            <th>Amplitudine</th>
                            <th>Trend</th>
                            <th>Scor anomalie</th>
                            <th>Interpretare</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cluster_rows}
                    </tbody>
                </table>
            </div>
        </section>

        {figure_card(
            fig_cluster_map,
            "Hartă clustere pixeli",
            "Fiecare pixel valid este mapat pe baza comportamentului temporal dominant.",
            section_id="cluster_map",
        )}

        {figure_card(
            fig_risk_map,
            "Hartă risc/anomalie temporală",
            "Zonele luminoase indică pixeli cu comportament temporal mai atipic conform Isolation Forest.",
            section_id="risk_map",
        )}

        {figure_card(
            cluster_profile_fig,
            "Profil temporal mediu pe cluster",
            "Graficul arată evoluția medie a indicelui pentru fiecare cluster.",
            section_id="cluster_profiles",
        )}

        {figure_card(
            fig_pca,
            "PCA 3D",
            "Fiecare punct reprezintă o fereastră temporală extrasă dintr-un pixel.",
            section_id="pca_3d",
        )}

        {tsne_html}

        {umap_html}

        {dtw_html}

        {amss_html}

        {figure_card(
            fig_pixel_compare,
            "Pixel de urmărit vs pixel reprezentativ",
            "Comparație temporală între un pixel cu scor ridicat de anomalie și un pixel reprezentativ.",
            section_id="dtw_pixel",
        )}
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

    description = INDEX_DESCRIPTIONS.get(
    selected_index,
    "Indice spectral utilizat în analiza vegetației."
)

    fig = go.Figure()

    for roi in df["roi"].unique():

        sub = df[df["roi"] == roi].sort_values("date")

        series = pd.Series(
            sub["value"].values,
            index=sub["date"]
        )

        smooth = smooth_series(series)

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=f"{roi} raw",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=smooth.index,
                y=smooth.values,
                mode="lines",
                name=f"{roi} trend",
                line=dict(width=4),
            )
        )

    fig.update_layout(
        title=f"Serie temporală {selected_index} – ROI1 vs ROI2",
        xaxis_title="Data",
        yaxis_title=selected_index,
    )

    stats = df.groupby("roi")["value"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()

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
    Această pagină extinde analiza NDVI către mai mulți indici spectrali
    utilizați în monitorizarea vegetației și umidității, precum EVI,
    SAVI, GNDVI, NDMI și AVI.
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
            "Seria temporală este obținută prin calculul mediei tuturor pixelilor din ROI pentru fiecare moment temporal din stack-ul multispectral.",
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
        import plotly.express as px
    except Exception:
        return render_template(
            "base.html",
            title="Analiză cross-index",
            nav_html=render_nav(request.path),
            content="""
            <section class="card reveal active">
              <h1>Cross-Index Analysis</h1>
              <p class="muted">
                Pentru această pagină este necesar pachetul plotly.
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
            title="Analiză cross-index",
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
        title="Analiză cross-index",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
  <h1>Analiză comparativă între indici spectrali</h1>

  <p class="muted">
    Această pagină compară indicii spectrali calculați pentru același ROI.
    Accentul este pus pe corelații și pe identificarea perechilor de indici
    care au evoluții temporale similare sau complementare.
  </p>

  <div class="method-box">
    <strong>Scopul analizei:</strong><br>
    Analiza cross-index permite observarea relațiilor dintre indicii
    spectrali și identificarea indicilor care descriu comportamente
    similare sau complementare ale vegetației.
  </div>

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
            complementari. 
          </div>
        </section>
        """,
    )




@main_bp.route("/methodology")
def methodology_page():
    df = load_ndvi()

    available_indices = sorted(
        df["index"].dropna().str.upper().unique().tolist()
    )

    available_rois = sorted(
        df["roi"].dropna().str.lower().unique().tolist()
    )

    if df.empty:
        global_date_min = "n/a"
        global_date_max = "n/a"
        total_observations = 0
    else:
        global_date_min = df["date"].min().strftime("%Y-%m-%d")
        global_date_max = df["date"].max().strftime("%Y-%m-%d")
        total_observations = len(df)

    summary_rows = ""

    if not df.empty:
        grouped = (
            df
            .assign(
                index_name=df["index"].str.upper(),
                roi_name=df["roi"].str.upper()
            )
            .groupby(["index_name", "roi_name"])
            .agg(
                observations=("value", "count"),
                date_min=("date", "min"),
                date_max=("date", "max"),
                mean_value=("value", "mean"),
                min_value=("value", "min"),
                max_value=("value", "max"),
            )
            .reset_index()
            .sort_values(["index_name", "roi_name"])
        )

        for _, row in grouped.iterrows():
            summary_rows += f"""
            <tr>
                <td>{row["index_name"]}</td>
                <td>{row["roi_name"]}</td>
                <td>{int(row["observations"])}</td>
                <td>{row["date_min"].strftime("%Y-%m-%d")} – {row["date_max"].strftime("%Y-%m-%d")}</td>
                <td>{float(row["mean_value"]):.4f}</td>
                <td>{float(row["min_value"]):.4f}</td>
                <td>{float(row["max_value"]):.4f}</td>
            </tr>
            """

    if not summary_rows:
        summary_rows = """
        <tr>
            <td colspan="7">Nu există date disponibile pentru sumar.</td>
        </tr>
        """

    indices_list = ", ".join(available_indices) if available_indices else "n/a"
    rois_list = ", ".join([roi.upper() for roi in available_rois]) if available_rois else "n/a"

    return render_template(
        "base.html",
        title="Metodologie",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h1>Metodologia aplicației</h1>
            <p class="muted">
                Această pagină explică modul în care aplicația transformă datele satelitare în rezultate
                interpretabile: serii temporale, componente statistice, hărți de risc și prognoze.
            </p>

            <div class="method-box">
                <strong>Obiectiv:</strong><br>
                Metodologia urmărește analiza evoluției vegetației în timp, prin combinarea indicilor
                spectrali cu tehnici de analiză temporală, machine learning nesupervizat și forecast.
            </div>
        </section>

        <section class="card reveal active">
            <h2>Date analizate</h2>

            <div class="method-box">
                <strong>Indici spectrali disponibili:</strong><br>
                {indices_list}
                <br><br>
                <strong>Regiuni de interes disponibile:</strong><br>
                {rois_list}
                <br><br>
                <strong>Interval temporal global:</strong><br>
                {global_date_min} – {global_date_max}
                <br><br>
                <strong>Total observații agregate:</strong><br>
                {total_observations}
            </div>

            <div class="table-wrap">
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Indice</th>
                            <th>ROI</th>
                            <th>Observații</th>
                            <th>Perioadă</th>
                            <th>Medie</th>
                            <th>Minim</th>
                            <th>Maxim</th>
                        </tr>
                    </thead>
                    <tbody>
                        {summary_rows}
                    </tbody>
                </table>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Pipeline metodologic</h2>

            <div class="pipeline">
                <div class="pipeline-step">
                    <span>1</span>
                    <p>Încărcarea datelor satelitare</p>
                </div>

                <div class="pipeline-step">
                    <span>2</span>
                    <p>Calculul indicilor spectrali</p>
                </div>

                <div class="pipeline-step">
                    <span>3</span>
                    <p>Agregarea temporală pe ROI</p>
                </div>

                <div class="pipeline-step">
                    <span>4</span>
                    <p>Analiza temporală și detectarea anomaliilor</p>
                </div>

                <div class="pipeline-step">
                    <span>5</span>
                    <p>ML pe pixeli și forecast</p>
                </div>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Etapele metodei</h2>

            <div class="method-box">
                <strong>1. Date satelitare și indici spectrali</strong><br>
                Datele sunt organizate pe regiuni de interes și pe indici spectrali. Fiecare indice
                oferă o perspectivă diferită asupra vegetației sau a suprafeței analizate.
            </div>

            <div class="method-box">
                <strong>2. Analiza temporală</strong><br>
                Seriile sunt analizate pentru evidențierea trendului, sezonalității, staționarității
                și valorilor anomale. Descompunerea STL separă seria în trend, componentă sezonieră
                și reziduu.
            </div>

            <div class="method-box">
                <strong>3. Machine Learning pe pixeli</strong><br>
                Fiecare pixel este tratat ca o semnătură temporală. Din această semnătură sunt extrase
                caracteristici, apoi pixelii sunt grupați prin K-Means. Isolation Forest este folosit
                pentru identificarea comportamentelor temporale atipice.
            </div>

            <div class="method-box">
                <strong>4. Reducere dimensională și similaritate</strong><br>
                PCA, t-SNE și UMAP sunt utilizate pentru vizualizarea semnăturilor temporale într-un
                spațiu redus. DTW compară similaritatea dintre profilele temporale ale clusterelor.
            </div>

            <div class="method-box">
                <strong>5. Forecast</strong><br>
                Modelele ARIMA/SARIMA și LSTM estimează evoluția viitoare a seriei. SARIMA este potrivit
                pentru sezonalitate explicită, iar LSTM este folosit ca metodă comparativă de tip deep learning.
            </div>
        </section>

        <section class="card reveal active">
            <h2>Interpretare metodologică</h2>

            <div class="method-box">
                <strong>ML nesupervizat:</strong><br>
                Clusterele și scorurile de anomalie nu reprezintă etichete reale ale terenului. Ele indică
                diferențe statistice în evoluția temporală a pixelilor și trebuie interpretate ca suport
                pentru analiză, nu ca diagnostic absolut.
            </div>

            <div class="method-box">
                <strong>Forecast:</strong><br>
                Prognozele sunt mai relevante pe termen scurt și mediu. Pe orizonturi lungi, diferențele
                dintre SARIMA și LSTM trebuie interpretate exploratoriu.
            </div>

            <div class="method-box">
                <strong>Valoarea proiectului:</strong><br>
                Aplicația unește teledetecția, analiza seriilor temporale, machine learning-ul nesupervizat
                și predicția într-un flux coerent pentru monitorizarea vegetației.
            </div>
        </section>
        """,
    )


