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
    load_indices_dataframe,
    INDEX_DESCRIPTIONS,
    smooth_series,
)
from services.dataset_service import (
    DEMO_DATASET_ID,
    build_dataset_options_html,
    get_dataset_display_name,
    get_dataset_rois,
    get_dataset_record,
    normalize_dataset_id,
    has_pixel_level_data,
    read_dataset_status,
    list_datasets,
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




def _catalog_status_badge(status: str) -> str:
    normalized = str(status or "unknown").strip().lower()
    meta = {
        "completed": ("status-completed", "Finalizat"),
        "processing": ("status-processing", "În procesare"),
        "uploaded": ("status-uploaded", "Încărcat"),
        "failed": ("status-failed", "Eșuat"),
        "demo": ("status-completed", "Demo"),
    }
    css_class, label = meta.get(normalized, ("status-unknown", normalized or "Necunoscut"))
    return f"""
    <span class="dataset-status-pill {css_class}">
        <span class="dataset-status-dot">●</span>
        <span>{label}</span>
    </span>
    """


def _catalog_dataset_cards() -> str:
    cards = ""

    for dataset in list_datasets(include_demo=True):
        dataset_id = dataset.get("dataset_id", DEMO_DATASET_ID)
        display_name = dataset.get("display_name", dataset_id)
        status = dataset.get("status", "demo" if dataset_id == DEMO_DATASET_ID else "uploaded")
        input_type = dataset.get("input_type", dataset.get("original_input_type", "-"))
        rows = dataset.get("rows", "-")
        rois = dataset.get("rois") or (["roi1", "roi2"] if dataset_id == DEMO_DATASET_ID else [])
        indices = dataset.get("indices") or []

        default_roi = rois[0] if rois else ("roi1" if dataset_id == DEMO_DATASET_ID else "parcela1")
        default_index = indices[0] if indices else "NDVI"

        rois_text = ", ".join(str(x).upper() for x in rois) if rois else "-"
        indices_text = ", ".join(str(x).upper() for x in indices) if indices else "-"

        ml_link = (
            f"/ml-features?dataset={dataset_id}&index={default_index}"
            f"&roi={default_roi}&pixels=500"
        )
        spectral_link = (
            f"/spectral-indices?dataset={dataset_id}&index={default_index}&roi={default_roi}"
        )
        temporal_link = (
            f"/stationarity?dataset={dataset_id}&index={default_index}&roi={default_roi}"
        )
        forecast_link = (
            f"/forecast-arima?dataset={dataset_id}&index={default_index}&roi={default_roi}"
        )
        json_link = (
            f"/api/series?dataset={dataset_id}&index={default_index}&roi={default_roi}"
        )

        cards += f"""
        <article class="catalog-dataset-card">
            <div class="dataset-card-header">
                <div>
                    <span class="panel-kicker">Dataset</span>
                    <h3>{display_name}</h3>
                    <p class="muted small-muted">{dataset_id}</p>
                </div>
                {_catalog_status_badge(status)}
            </div>

            <div class="dataset-card-meta-grid">
                <div><span>Tip</span><strong>{input_type}</strong></div>
                <div><span>Rânduri</span><strong>{rows}</strong></div>
                <div><span>ROI-uri</span><strong>{rois_text}</strong></div>
                <div><span>Indici</span><strong>{indices_text}</strong></div>
            </div>

            <div class="dataset-actions catalog-actions">
                <a class="btn-link" href="{spectral_link}">Analiză</a>
                <a class="btn-link secondary" href="{temporal_link}">Temporal</a>
                <a class="btn-link secondary" href="{ml_link}">ML</a>
                <a class="btn-link secondary" href="{forecast_link}">Forecast</a>
                <a class="btn-link secondary" href="{json_link}">JSON</a>
            </div>
        </article>
        """

    return cards


@main_bp.route("/series-catalog")
def series_catalog():
    dataset_cards = _catalog_dataset_cards()

    return render_template(
        "base.html",
        title="Catalog serii și dataseturi",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active catalog-hero-card">
            <div class="card-top-line"></div>
            <div class="section-heading horizontal-heading">
                <div>
                    <h1>Catalog serii și dataseturi</h1>
                </div>
                <a class="btn-link" href="/datasets">Gestionează dataseturi</a>
            </div>

            <div class="method-box">
                <strong>Flux recomandat:</strong><br>
                Încarcă sau selectează un dataset, verifică indicii și ROI-urile disponibile, apoi continuă cu
                analiza temporală, hărțile ML și prognoza. Dataseturile încărcate de utilizator sunt integrate
                în aceleași module ca datasetul demonstrativ.
            </div>
        </section>

        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h2>Fluxul principal</h2>

            <div class="pipeline">
                <div class="pipeline-step"><span>1</span><p>Dataseturi</p></div>
                <div class="pipeline-step"><span>2</span><p>Indici și ROI-uri</p></div>
                <div class="pipeline-step"><span>3</span><p>Analiză temporală</p></div>
                <div class="pipeline-step"><span>4</span><p>ML pe pixeli</p></div>
                <div class="pipeline-step"><span>5</span><p>Forecast</p></div>
            </div>
        </section>

        <section class="card reveal active">
            <div class="card-top-line"></div>
            <div class="section-heading horizontal-heading">
                <div>
                    <h2>Dataseturi disponibile</h2>
                    <p class="muted">
                        Cardurile de mai jos folosesc metadatele din platformă: status, ROI-uri, indici și număr de rânduri.
                    </p>
                </div>
                <a class="btn-link secondary" href="/datasets">Încarcă dataset nou</a>
            </div>
            <div class="catalog-dataset-grid">
                {dataset_cards}
            </div>
        </section>

        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h2>Modulele aplicației</h2>

            <div class="catalog-grid">
                <div class="catalog-card">
                    <div class="badge blue">Date</div>
                    <h2>Date și ROI-uri</h2>
                    <p class="muted">Încărcarea, inspectarea și administrarea dataseturilor utilizator.</p>
                    <ul>
                        <li><a href="/datasets">Dataseturi utilizator</a></li>
                        <li><a href="/roi">ROI demonstrative</a></li>
                    </ul>
                </div>

                <div class="catalog-card">
                    <div class="badge green">Indici</div>
                    <h2>Indici spectrali</h2>
                    <p class="muted">Vizualizare pentru NDVI, NDMI, SAVI, AVI, EVI și GNDVI.</p>
                    <ul>
                        <li><a href="/spectral-indices">Analiza indicilor</a></li>
                        <li><a href="/cross-index-analysis">Analiză Cross-Index</a></li>
                    </ul>
                </div>

                <div class="catalog-card">
                    <div class="badge cyan">Serii</div>
                    <h2>Serii sintetice</h2>
                    <p class="muted">Exemple controlate pentru staționaritate, trend, sezonalitate și zgomot.</p>
                    <ul>
                        <li><a href="/synthetic">Catalog serii sintetice</a></li>
                        <li><a href="/temperature-demo">Temperatură demonstrativă</a></li>
                    </ul>
                </div>

                <div class="catalog-card">
                    <div class="badge teal">Temporal</div>
                    <h2>Analiză temporală</h2>
                    <p class="muted">Staționaritate, STL, trend, sezonalitate și detecția anomaliilor.</p>
                    <ul>
                        <li><a href="/stationarity">Staționaritate</a></li>
                        <li><a href="/decompose">Descompunere STL</a></li>
                        <li><a href="/anomalies">Anomalii</a></li>
                    </ul>
                </div>

                <div class="catalog-card highlight-card">
                    <div class="badge red">ML</div>
                    <h2>Machine Learning pe pixeli</h2>
                    <p class="muted">K-Means, Isolation Forest, PCA, t-SNE, UMAP, DTW și hărți de risc.</p>
                    <ul><li><a href="/ml-features">Deschide modulul ML</a></li></ul>
                </div>

                <div class="catalog-card">
                    <div class="badge purple">Forecast</div>
                    <h2>Prognoză temporală</h2>
                    <p class="muted">Estimarea evoluției viitoare prin ARIMA/SARIMA și LSTM.</p>
                    <ul>
                        <li><a href="/forecast-arima">Forecast ARIMA / SARIMA</a></li>
                        <li><a href="/forecast-lstm">Forecast LSTM</a></li>
                    </ul>
                </div>

                <div class="catalog-card">
                    <div class="badge gray">Metodologie</div>
                    <h2>Metodologie</h2>
                    <p class="muted">Descrierea pipeline-ului, ipotezelor, metodelor și arhitecturii cloud.</p>
                    <ul>
                        <li><a href="/methodology">Deschide metodologia</a></li>
                        <li><a href="/debug">Debug date</a></li>
                    </ul>
                </div>
            </div>
        </section>
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
            <a class="btn-link" href="{info["route"]}">Vezi seria</a>
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



def build_farmer_cluster_interpretation(row, selected_index):
    """
    Transformă metricele unui cluster într-o explicație practică.

    Interpretarea este orientativă și este folosită pentru prioritizarea
    verificărilor în teren, nu pentru diagnostic agronomic final.
    """

    mean_value = float(row.get("mean", 0))
    amplitude = float(row.get("amplitude", 0))
    risk_score = float(row.get("risk_score", 0))
    trend = str(row.get("trend", "")).lower()

    index_name = selected_index.upper()

    if index_name in ["NDVI", "EVI", "GNDVI", "SAVI", "AVI"]:
        low_label = "nivel redus de vegetație"
        good_label = "vegetație mai viguroasă"
        field_check = (
            "verificare în teren pentru stres vegetal, sol expus, fertilizare slabă "
            "sau zone cu dezvoltare mai redusă"
        )
    elif index_name == "NDMI":
        low_label = "nivel redus de umiditate"
        good_label = "umiditate mai bună a vegetației"
        field_check = (
            "verificare în teren pentru posibil deficit de apă, irigare neuniformă "
            "sau stres hidric"
        )
    else:
        low_label = "nivel redus al indicelui"
        good_label = "nivel mai ridicat al indicelui"
        field_check = "verificare în teren pentru diferențe locale față de restul parcelei"

    if risk_score >= 0:
        priority = "ridicată"
        priority_class = "priority-high"
        priority_text = (
            "zonă prioritară pentru verificare, deoarece are comportament mai atipic "
            "față de restul regiunii"
        )
    elif risk_score >= -0.08:
        priority = "medie"
        priority_class = "priority-medium"
        priority_text = "zonă care merită monitorizată, fără a indica obligatoriu o problemă critică"
    else:
        priority = "scăzută"
        priority_class = "priority-low"
        priority_text = "zonă relativ stabilă, cu comportament apropiat de restul regiunii"

    if mean_value < 0.25:
        meaning = f"Clusterul indică {low_label}."
    elif mean_value < 0.45:
        meaning = "Clusterul indică o stare intermediară a suprafeței analizate."
    else:
        meaning = f"Clusterul indică {good_label}."

    if amplitude > 0.25:
        seasonality = "variație sezonieră puternică"
    elif amplitude > 0.10:
        seasonality = "variație sezonieră moderată"
    else:
        seasonality = "variație sezonieră redusă"

    if "desc" in trend or "negativ" in trend:
        trend_text = "evoluția temporală sugerează o posibilă scădere în timp"
    elif "asc" in trend or "pozitiv" in trend:
        trend_text = "evoluția temporală sugerează o posibilă creștere în timp"
    else:
        trend_text = "nu se observă o direcție clară a evoluției"

    return {
        "meaning": meaning,
        "priority": priority,
        "priority_class": priority_class,
        "priority_text": priority_text,
        "seasonality": seasonality,
        "trend_text": trend_text,
        "recommended_action": field_check,
    }


@main_bp.route("/ml-features")
def ml_features_page():
    selected_dataset = normalize_dataset_id(request.args.get("dataset", DEMO_DATASET_ID))

    user_payload = None

    if selected_dataset != DEMO_DATASET_ID:
        available_indices = list_indices(dataset_id=selected_dataset)
        if not available_indices:
            available_indices = ["NDVI"]

        selected_index = request.args.get("index", available_indices[0]).upper()
        if selected_index not in available_indices:
            selected_index = available_indices[0]

        available_rois = [str(r).lower() for r in get_dataset_rois(selected_dataset)]
        if not available_rois:
            available_rois = ["parcela1"]

        roi = request.args.get("roi", available_rois[0]).lower()
        if roi not in available_rois:
            roi = available_rois[0]

        try:
            pixel_count = int(request.args.get("pixels", "500"))
        except Exception:
            pixel_count = 500
        if pixel_count not in DEFAULT_PIXEL_COUNTS:
            pixel_count = min(DEFAULT_PIXEL_COUNTS, key=lambda x: abs(x - pixel_count))

    else:
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
    roi_values_for_options = get_dataset_rois(selected_dataset) if selected_dataset != DEMO_DATASET_ID else ["roi1", "roi2"]

    for roi_name in roi_values_for_options:
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
            dataset_id=selected_dataset,
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
                    <label><strong>Dataset:</strong></label><br>
                    <select name="dataset" onchange="this.form.submit()" class="select-input">
                        {build_dataset_options_html(selected_dataset)}
                    </select>

                    <br><br>

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
        title="PCA 3D – vizualizare clustere K-Means",
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
    farmer_rows = ""

    for row in payload["cluster_summary"]:
        farmer_info = build_farmer_cluster_interpretation(
            row,
            selected_index,
        )

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

        farmer_rows += f"""
        <tr>
            <td><strong>Cluster {row["cluster"]}</strong></td>
            <td>{farmer_info["meaning"]}</td>
            <td>
                <span class="priority-badge {farmer_info["priority_class"]}">
                    {farmer_info["priority"].capitalize()}
                </span>
            </td>
            <td>{farmer_info["trend_text"]}; {farmer_info["seasonality"]}.</td>
            <td>{farmer_info["recommended_action"]}.</td>
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
                <label><strong>Dataset:</strong></label><br>
                <select name="dataset" onchange="this.form.submit()" class="select-input">
                    {build_dataset_options_html(selected_dataset)}
                </select>

                <br><br>

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
                Pixeli folosiți la antrenarea modelului: <strong>{metadata["pixel_count"]}</strong><br>
                Pixeli valizi afișați pe hartă: <strong>{metadata["mapped_pixels"]}</strong><br>
                Ferestre temporale analizate: <strong>{metadata["windows_extracted"]}</strong>
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

        <section class="card reveal active farmer-guide-card">
            <div class="section-heading">
                <span class="section-kicker">Interpretare practică</span>
                <h2>Cum se citește harta?</h2>
                <p class="muted">
                    Rezultatele nu reprezintă un diagnostic agronomic final, ci un mod de a prioritiza
                    zonele care merită verificate în teren. Pentru un utilizator practic, harta indică
                    unde apar diferențe temporale, nu cauza exactă a acestor diferențe.
                </p>
            </div>

            <div class="farmer-guide-grid">
                <div class="farmer-guide-item">
                    <span class="guide-number">1</span>
                    <h3>Harta de clustere</h3>
                    <p>
                        Fiecare culoare reprezintă un grup de pixeli care au avut o evoluție temporală
                        asemănătoare a indicelui <strong>{selected_index}</strong>. Clusterele nu sunt
                        culturi sau clase de teren etichetate manual, ci tipare statistice identificate automat.
                    </p>
                </div>

                <div class="farmer-guide-item">
                    <span class="guide-number">2</span>
                    <h3>Harta de risc</h3>
                    <p>
                        Zonele cu scor mai ridicat indică pixeli cu comportament temporal mai atipic.
                        Aceste zone ar trebui verificate primele, deoarece pot semnala stres vegetal,
                        deficit de umiditate, sol expus sau schimbări față de comportamentul dominant.
                    </p>
                </div>

                <div class="farmer-guide-item">
                    <span class="guide-number">3</span>
                    <h3>Hover pe hartă</h3>
                    <p>
                        Valorile afișate la trecerea cu mouse-ul peste hartă descriu poziția pixelului
                        în grila imaginii, clusterul asociat și/sau scorul de anomalie. În forma actuală,
                        poziția este exprimată în coordonate de imagine, nu ca GPS.
                    </p>
                </div>
            </div>

            <div class="method-box decision-note">
                <strong>Important:</strong><br>
                Platforma oferă suport decizional și identifică zone de verificat. Nu recomandă automat
                tratamente, irigare sau fertilizare, deoarece aceste decizii necesită confirmare în teren
                și informații suplimentare despre cultură, sol și lucrări agricole.
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

        <section class="card reveal active farmer-table-card">
            <div class="section-heading">
                <h2>Interpretare practică pe clustere</h2>
                <p class="muted">
                    Tabelul traduce rezultatele tehnice în observații practice. Recomandările indică zone
                    de verificat, nu acțiuni agricole automate.
                </p>
            </div>

            <div class="table-wrap">
                <table class="stats-table farmer-table">
                    <thead>
                        <tr>
                            <th>Cluster</th>
                            <th>Ce poate indica</th>
                            <th>Prioritate</th>
                            <th>Comportament temporal</th>
                            <th>Acțiune recomandată</th>
                        </tr>
                    </thead>
                    <tbody>
                        {farmer_rows}
                    </tbody>
                </table>
            </div>
        </section>

        {figure_card(
            fig_cluster_map,
            "Hartă clustere pixeli",
            "Fiecare culoare reprezintă un grup de pixeli cu evoluție temporală asemănătoare. Harta ajută la delimitarea zonelor cu comportament diferit în interiorul regiunii analizate.",
            section_id="cluster_map",
        )}

        {figure_card(
            fig_risk_map,
            "Hartă risc/anomalie temporală",
            "Zonele cu scor mai ridicat indică pixeli care se abat de la comportamentul temporal dominant și ar trebui prioritizați pentru verificare în teren.",
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
            "PCA 3D – rezultat K-Means",
            "PCA proiectează semnăturile temporale într-un spațiu redus. Culorile reprezintă clusterele atribuite prin K-Means.",
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




@main_bp.route("/api/series")
def api_series():
    selected_dataset = normalize_dataset_id(request.args.get("dataset", DEMO_DATASET_ID))
    selected_index = request.args.get("index", "NDVI").upper()
    selected_roi = request.args.get("roi") or request.args.get("site") or "roi1"
    selected_roi = selected_roi.lower()

    try:
        df = load_index_dataframe(selected_index, dataset_id=selected_dataset)
        df = df[df["roi"].astype(str).str.lower() == selected_roi].copy()
        if df.empty:
            return jsonify({
                "status": "error",
                "message": f"Nu există date pentru {selected_index} - {selected_roi}.",
                "dataset": selected_dataset,
            }), 404

        df = df.sort_values("date")
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "roi": str(row["roi"]),
                "index": str(row["index"]).upper(),
                "value": float(row["value"]),
            })

        return jsonify({
            "status": "ok",
            "dataset": selected_dataset,
            "roi": selected_roi,
            "index": selected_index,
            "count": len(rows),
            "data": rows,
        })
    except Exception as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
            "dataset": selected_dataset,
        }), 500

@main_bp.route("/spectral-indices")
def spectral_indices_page():
    selected_dataset = normalize_dataset_id(request.args.get("dataset", DEMO_DATASET_ID))
    dataset_name = get_dataset_display_name(selected_dataset)
    dataset_options = build_dataset_options_html(selected_dataset)
    available_indices = list_indices(dataset_id=selected_dataset)

    selected_index = request.args.get("index", "NDVI").upper()
    if selected_index not in available_indices:
        selected_index = available_indices[0] if available_indices else "NDVI"

    try:
        df = load_index_dataframe(selected_index, dataset_id=selected_dataset)
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

    for roi_name in df["roi"].unique():
        sub = df[df["roi"] == roi_name].sort_values("date")
        series = pd.Series(sub["value"].values, index=sub["date"])
        smooth = smooth_series(series)

        fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=f"{roi_name} raw"))
        fig.add_trace(go.Scatter(x=smooth.index, y=smooth.values, mode="lines", name=f"{roi_name} trend", line=dict(width=4)))

    fig.update_layout(
        title=f"Serie temporală {selected_index} – {dataset_name}",
        xaxis_title="Data",
        yaxis_title=selected_index,
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
            Această pagină permite analiza indicilor spectrali pentru ROI-urile demonstrative sau pentru
            dataset-uri încărcate de utilizator.
          </p>

          <form method="get" class="method-box">
            <label><strong>Dataset:</strong></label><br><br>
            <select name="dataset" onchange="this.form.submit()" class="select-input">
              {dataset_options}
            </select>

            <br><br>

            <label for="index"><strong>Selectează indicele spectral:</strong></label><br><br>
            <select name="index" id="index" onchange="this.form.submit()" class="select-input">
              {options_html}
            </select>
          </form>

          <div class="method-box">
            <strong>Dataset:</strong> {dataset_name}<br><br>
            <strong>{selected_index}:</strong><br>
            {description}<br><br>
            <a class="btn-link secondary" href="/datasets/{selected_dataset}/download">Descarcă CSV dataset</a>
          </div>
        </section>

        {figure_card(
            fig,
            f"{selected_index} – serie temporală",
            "Seria temporală este obținută prin calculul mediei valorilor pentru fiecare ROI și moment temporal.",
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
              <p class="muted">Pentru această pagină este necesar pachetul plotly.</p>
            </section>
            """,
        )

    selected_dataset = normalize_dataset_id(request.args.get("dataset", DEMO_DATASET_ID))
    dataset_name = get_dataset_display_name(selected_dataset)
    dataset_options = build_dataset_options_html(selected_dataset)
    available_rois = get_dataset_rois(selected_dataset)

    selected_roi = request.args.get("roi", available_rois[0] if available_rois else "roi1").lower()
    if selected_roi not in available_rois:
        selected_roi = available_rois[0] if available_rois else "roi1"

    wide_df = build_indices_wide_dataframe(roi=selected_roi, dataset_id=selected_dataset)

    if wide_df.empty or wide_df.shape[1] < 2:
        return render_template(
            "base.html",
            title="Analiză cross-index",
            nav_html=render_nav(request.path),
            content=f"""
            <section class="card reveal active">
              <h1>Cross-Index Analysis</h1>
              <p class="muted">Nu există suficiente date pentru analiza comparativă între indici.</p>
              <a class="btn-link secondary" href="/datasets">Încarcă dataset</a>
            </section>
            """,
        )

    roi_options = ""
    for roi_name in available_rois:
        selected = "selected" if roi_name == selected_roi else ""
        roi_options += f"<option value='{roi_name}' {selected}>{roi_name.upper()}</option>"

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
          </p>

          <form method="get" class="method-box">
            <label><strong>Dataset:</strong></label><br><br>
            <select name="dataset" onchange="this.form.submit()" class="select-input">
              {dataset_options}
            </select>

            <br><br>

            <label for="roi"><strong>Selectează ROI:</strong></label><br><br>
            <select name="roi" id="roi" onchange="this.form.submit()" class="select-input">
              {roi_options}
            </select>
          </form>

          <div class="method-box">
            <strong>Dataset:</strong> {dataset_name}<br>
            <a class="btn-link secondary" href="/datasets/{selected_dataset}/download">Descarcă CSV dataset</a>
          </div>
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
              <tbody>{table_rows}</tbody>
            </table>
          </div>
        </section>

        <section class="card reveal active">
          <h2>Interpretare automată</h2>
          <div class="method-box">
            <strong>Cea mai mare similaritate:</strong><br>
            Perechea <strong>{strongest['index_a']} – {strongest['index_b']}</strong>
            are corelația <strong>{round(float(strongest['corr']), 4)}</strong>.
          </div>
          <div class="method-box">
            <strong>Cea mai mică similaritate:</strong><br>
            Perechea <strong>{weakest['index_a']} – {weakest['index_b']}</strong>
            are corelația <strong>{round(float(weakest['corr']), 4)}</strong>.
          </div>
        </section>
        """,
    )





@main_bp.route("/methodology")
def methodology_page():
    selected_dataset = normalize_dataset_id(request.args.get("dataset", DEMO_DATASET_ID))
    dataset_name = get_dataset_display_name(selected_dataset)
    dataset_options = build_dataset_options_html(selected_dataset)

    df = load_indices_dataframe(dataset_id=selected_dataset)

    available_indices = sorted(df["index"].dropna().str.upper().unique().tolist())
    available_rois = sorted(df["roi"].dropna().str.lower().unique().tolist())

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
            .assign(index_name=df["index"].str.upper(), roi_name=df["roi"].str.upper())
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
        <tr><td colspan="7">Nu există date disponibile pentru sumar.</td></tr>
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

            <form method="get" class="method-box">
                <label><strong>Dataset:</strong></label><br><br>
                <select name="dataset" onchange="this.form.submit()" class="select-input">
                    {dataset_options}
                </select>
            </form>

            <div class="method-box">
                <strong>Dataset curent:</strong> {dataset_name}<br>
                <a class="btn-link secondary" href="/datasets/{selected_dataset}/download">Descarcă CSV dataset</a>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Date analizate</h2>
            <div class="method-box">
                <strong>Indici spectrali disponibili:</strong><br>{indices_list}<br><br>
                <strong>Regiuni de interes disponibile:</strong><br>{rois_list}<br><br>
                <strong>Interval temporal global:</strong><br>{global_date_min} – {global_date_max}<br><br>
                <strong>Total observații agregate:</strong><br>{total_observations}
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
                    <tbody>{summary_rows}</tbody>
                </table>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Pipeline metodologic</h2>
            <div class="pipeline">
                <div class="pipeline-step"><span>1</span><p>Încărcarea datelor</p></div>
                <div class="pipeline-step"><span>2</span><p>Validare CSV / ROI</p></div>
                <div class="pipeline-step"><span>3</span><p>Analiză spectrală</p></div>
                <div class="pipeline-step"><span>4</span><p>Analiză temporală</p></div>
                <div class="pipeline-step"><span>5</span><p>ML pe pixeli și forecast</p></div>
            </div>
        </section>

        <section class="card reveal active">
            <h2>Interpretare metodologică</h2>
            <div class="method-box">
                <strong>Dataset-uri utilizator:</strong><br>
                În faza 1, dataset-urile încărcate de utilizator sunt CSV-uri agregate la nivel de ROI.
                Acestea pot fi folosite pentru analiza indicilor, Cross-Index și forecast. Pentru hărți ML
                la nivel de pixel este necesar un dataset pixel-level sau o procesare precompute separată.
            </div>
            <div class="method-box">
                <strong>ML nesupervizat:</strong><br>
                Clusterele și scorurile de anomalie nu reprezintă etichete reale ale terenului. Ele indică
                diferențe statistice în evoluția temporală a pixelilor și trebuie interpretate ca suport
                pentru analiză, nu ca diagnostic absolut.
            </div>
        </section>
        """,
    )

