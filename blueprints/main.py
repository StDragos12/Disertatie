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

    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score,
        )
        import plotly.express as px

        try:
            import umap
            umap_available = True
        except Exception:
            umap_available = False

    except Exception:
        return render_template(
            "base.html",
            title="ML pe pixeli",
            nav_html=render_nav(request.path),
            content="""
            <section class="card reveal active">
              <h1>Analiză ML pe pixeli</h1>
              <p class="muted">
                Lipsesc dependențe ML. Verifică instalarea pachetelor scikit-learn, plotly, tslearn, umap-learn și scipy.
              </p>
            </section>
            """,
        )

    from services.indices_service import load_index_array

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

    max_pixels = 1200
    max_embedding_rows = 1200
    max_profile_pixels_per_cluster = 450

    window = 12
    step = 6
    n_clusters = 4

    arr = load_index_array(selected_index, roi)

    height, width, series_length = arr.shape

    flat_pixels = arr.reshape(
        -1,
        series_length
    )

    valid_fraction = np.isfinite(flat_pixels).mean(axis=1)
    valid_mask = valid_fraction >= 0.70

    valid_indices_all = np.where(valid_mask)[0]
    pixels_all = flat_pixels[valid_mask]

    if len(pixels_all) == 0:
        return render_template(
            "base.html",
            title="ML pe pixeli",
            nav_html=render_nav(request.path),
            content="""
            <section class="card reveal active">
              <h1>Analiză ML pe pixeli</h1>
              <p class="muted">
                Nu există pixeli validați pentru analiza ML.
              </p>
            </section>
            """,
        )

    total_valid_pixels = len(pixels_all)

    if total_valid_pixels < n_clusters:
        return render_template(
            "base.html",
            title="ML pe pixeli",
            nav_html=render_nav(request.path),
            content=f"""
            <section class="card reveal active">
              <h1>Analiză ML pe pixeli</h1>
              <p class="muted">
                Nu există suficienți pixeli validați pentru {n_clusters} clustere.
              </p>
            </section>
            """,
        )

    rng = np.random.default_rng(42)

    if len(pixels_all) > max_pixels:
        sampled_idx = rng.choice(
            len(pixels_all),
            max_pixels,
            replace=False
        )

        pixels = pixels_all[sampled_idx]
        valid_indices = valid_indices_all[sampled_idx]
    else:
        pixels = pixels_all
        valid_indices = valid_indices_all

    try:
        df_dates = load_index_dataframe(selected_index)
        df_dates = df_dates[df_dates["roi"].str.lower() == roi].copy()
        df_dates["date"] = pd.to_datetime(df_dates["date"], errors="coerce")

        unique_dates = (
            df_dates["date"]
            .dropna()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        if len(unique_dates) == series_length:
            dates = pd.DatetimeIndex(unique_dates)
        else:
            dates = pd.date_range(
                start="2017-01-01",
                periods=series_length,
                freq="MS"
            )

    except Exception:
        dates = pd.date_range(
            start="2017-01-01",
            periods=series_length,
            freq="MS"
        )

    def make_clean_series(values: np.ndarray) -> pd.Series:
        series = pd.Series(
            values.astype(float),
            index=dates
        )

        series = series.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if series.isna().mean() > 0.30:
            return pd.Series(dtype=float)

        series = series.interpolate(
            method="time"
        ).bfill().ffill()

        if series.isna().any():
            return pd.Series(dtype=float)

        return series.astype(float)

    all_rows = []
    clean_pixel_series = {}

    for local_pixel_id, values in enumerate(pixels):

        original_pixel_id = int(valid_indices[local_pixel_id])

        series = make_clean_series(values)

        if series.empty or len(series) < window:
            continue

        clean_pixel_series[original_pixel_id] = series

        for start_pos in range(
            0,
            len(series) - window + 1,
            step
        ):

            sub = series.iloc[
                start_pos:start_pos + window
            ]

            features = extract_features(sub)

            all_rows.append({
                "pixel_id": original_pixel_id,
                "window_start": sub.index[0].strftime("%Y-%m"),
                "window_end": sub.index[-1].strftime("%Y-%m"),
                "mean": features["mean"],
                "std": features["std"],
                "amplitude": features["amplitude"],
                "trend_slope": features["trend_slope"],
                "anomaly_count": features["anomaly_count"],
            })

    features_df = pd.DataFrame(all_rows)

    if features_df.empty or len(features_df) < n_clusters:
        return render_template(
            "base.html",
            title="ML pe pixeli",
            nav_html=render_nav(request.path),
            content="""
            <section class="card reveal active">
              <h1>Analiză ML pe pixeli</h1>
              <p class="muted">
                Nu există suficiente date după feature extraction pentru patru clustere.
              </p>
            </section>
            """,
        )

    feature_cols = [
        "mean",
        "std",
        "amplitude",
        "trend_slope",
        "anomaly_count",
    ]

    X = features_df[feature_cols].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    clusters = kmeans.fit_predict(X_scaled)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
        n_jobs=-1,
    )

    outlier_labels = iso.fit_predict(X_scaled)
    risk_scores = -iso.decision_function(X_scaled)

    pca = PCA(
        n_components=3,
        random_state=42
    )

    X_pca = pca.fit_transform(X_scaled)

    pca_df = features_df.copy()
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]
    pca_df["PC3"] = X_pca[:, 2]
    pca_df["cluster"] = clusters.astype(int)
    pca_df["cluster_label"] = [
        f"Cluster {cluster_id + 1}"
        for cluster_id in clusters
    ]
    pca_df["risk_score"] = risk_scores
    pca_df["outlier"] = outlier_labels

    silhouette = None
    calinski = None
    davies = None

    if len(np.unique(clusters)) > 1 and len(features_df) > n_clusters:
        try:
            metric_sample_size = min(
                1500,
                len(X_scaled)
            )

            if len(X_scaled) > metric_sample_size:
                metric_idx = rng.choice(
                    len(X_scaled),
                    metric_sample_size,
                    replace=False
                )
                X_metric = X_scaled[metric_idx]
                clusters_metric = clusters[metric_idx]
            else:
                X_metric = X_scaled
                clusters_metric = clusters

            silhouette = float(
                silhouette_score(
                    X_metric,
                    clusters_metric
                )
            )

            calinski = float(
                calinski_harabasz_score(
                    X_metric,
                    clusters_metric
                )
            )

            davies = float(
                davies_bouldin_score(
                    X_metric,
                    clusters_metric
                )
            )

        except Exception:
            silhouette = None
            calinski = None
            davies = None

    def build_vectorized_pixel_features(pixel_matrix: np.ndarray) -> pd.DataFrame:
        matrix = pixel_matrix.astype(float)
        matrix = np.where(np.isfinite(matrix), matrix, np.nan)

        row_mean = np.nanmean(matrix, axis=1)
        row_median = np.nanmedian(matrix, axis=1)
        row_std = np.nanstd(matrix, axis=1)
        row_min = np.nanmin(matrix, axis=1)
        row_max = np.nanmax(matrix, axis=1)

        filled = matrix.copy()
        nan_rows, nan_cols = np.where(~np.isfinite(filled))

        if len(nan_rows) > 0:
            filled[nan_rows, nan_cols] = row_mean[nan_rows]

        x_time = np.arange(matrix.shape[1], dtype=float)
        x_centered = x_time - x_time.mean()
        denom = np.sum(x_centered ** 2)

        y_centered = filled - row_mean[:, None]
        trend_slope = np.sum(y_centered * x_centered[None, :], axis=1) / denom

        sigma = np.where(row_std > 1e-8, row_std, np.nan)
        z_values = np.abs(matrix - row_median[:, None]) / sigma[:, None]
        anomaly_count = np.nansum(z_values > 3.0, axis=1)

        feature_df = pd.DataFrame({
            "mean": row_mean,
            "std": row_std,
            "amplitude": row_max - row_min,
            "trend_slope": trend_slope,
            "anomaly_count": anomaly_count,
        })

        feature_df = feature_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

        return feature_df

    map_features_df = build_vectorized_pixel_features(pixels_all)
    X_map_scaled = scaler.transform(map_features_df[feature_cols].astype(float))

    map_clusters = kmeans.predict(X_map_scaled)
    map_risk_scores = -iso.decision_function(X_map_scaled)

    cluster_grid = np.full(
        (height, width),
        np.nan
    )

    risk_grid = np.full(
        (height, width),
        np.nan
    )

    pixel_cluster_final = {}
    pixel_risk_avg = {}

    for pos, pixel_id in enumerate(valid_indices_all):
        pixel_id = int(pixel_id)

        row = pixel_id // width
        col = pixel_id % width

        assigned_cluster = int(map_clusters[pos])
        assigned_risk = float(map_risk_scores[pos])

        pixel_cluster_final[pixel_id] = assigned_cluster
        pixel_risk_avg[pixel_id] = assigned_risk

        cluster_grid[row, col] = assigned_cluster + 1
        risk_grid[row, col] = assigned_risk

    map_cluster_counts = (
        pd.Series(map_clusters)
        .value_counts()
        .reindex(range(n_clusters), fill_value=0)
        .to_dict()
    )

    cluster_summary = (
        pca_df
        .groupby("cluster")
        .agg(
            windows=("pixel_id", "count"),
            unique_pixels=("pixel_id", "nunique"),
            mean_value=("mean", "mean"),
            std_value=("std", "mean"),
            amplitude=("amplitude", "mean"),
            trend_slope=("trend_slope", "mean"),
            anomaly_count=("anomaly_count", "mean"),
            risk_score=("risk_score", "mean"),
        )
        .reindex(range(n_clusters))
        .reset_index()
        .sort_values("cluster")
    )

    cluster_summary["windows"] = cluster_summary["windows"].fillna(0).astype(int)
    cluster_summary["unique_pixels"] = cluster_summary["unique_pixels"].fillna(0).astype(int)

    numeric_cols = [
        "mean_value",
        "std_value",
        "amplitude",
        "trend_slope",
        "anomaly_count",
        "risk_score",
    ]

    for col in numeric_cols:
        cluster_summary[col] = cluster_summary[col].fillna(0.0)

    cluster_summary["mapped_pixels"] = cluster_summary["cluster"].map(
        lambda cluster_id: int(map_cluster_counts.get(int(cluster_id), 0))
    )

    def interpret_cluster(row):
        if int(row["mapped_pixels"]) == 0:
            return "cluster fără pixeli mapați"

        risk_q75 = cluster_summary["risk_score"].quantile(0.75)
        amp_q75 = cluster_summary["amplitude"].quantile(0.75)
        mean_q25 = cluster_summary["mean_value"].quantile(0.25)

        if row["risk_score"] >= risk_q75:
            return "zonă de urmărit / comportament atipic"

        if row["amplitude"] >= amp_q75:
            return "sezonalitate puternică"

        if row["mean_value"] <= mean_q25:
            return "vegetație redusă"

        if abs(row["trend_slope"]) <= 0.002:
            return "comportament stabil"

        if row["trend_slope"] > 0:
            return "tendință ascendentă"

        return "tendință descendentă"

    cluster_summary["interpretation"] = cluster_summary.apply(
        interpret_cluster,
        axis=1
    )

    table_rows = ""

    for _, row_data in cluster_summary.iterrows():
        table_rows += f"""
        <tr>
            <td>Cluster {int(row_data["cluster"]) + 1}</td>
            <td>{int(row_data["unique_pixels"])}</td>
            <td>{int(row_data["mapped_pixels"])}</td>
            <td>{int(row_data["windows"])}</td>
            <td>{row_data["mean_value"]:.4f}</td>
            <td>{row_data["amplitude"]:.4f}</td>
            <td>{row_data["trend_slope"]:.5f}</td>
            <td>{row_data["risk_score"]:.4f}</td>
            <td>{row_data["interpretation"]}</td>
        </tr>
        """

    dominant_cluster = cluster_summary.sort_values(
        "mapped_pixels",
        ascending=False
    ).iloc[0]

    risk_cluster = cluster_summary.sort_values(
        "risk_score",
        ascending=False
    ).iloc[0]

    seasonal_cluster = cluster_summary.sort_values(
        "amplitude",
        ascending=False
    ).iloc[0]

    low_vegetation_cluster = cluster_summary.sort_values(
        "mean_value",
        ascending=True
    ).iloc[0]

    metric_silhouette = "n/a" if silhouette is None else f"{silhouette:.3f}"
    metric_calinski = "n/a" if calinski is None else f"{calinski:.1f}"
    metric_davies = "n/a" if davies is None else f"{davies:.3f}"

    if silhouette is None:
        cluster_quality_text = "Calitatea separării nu a putut fi evaluată numeric pentru configurația curentă."
    elif silhouette >= 0.50:
        cluster_quality_text = "Separarea clusterelor este bună; grupele temporale sunt relativ distincte."
    elif silhouette >= 0.25:
        cluster_quality_text = "Separarea clusterelor este moderată; unele zone au comportamente apropiate."
    else:
        cluster_quality_text = "Separarea clusterelor este slabă spre moderată; rezultatele trebuie interpretate exploratoriu."

    farmer_recommendation = f"""
    Pentru interpretare practică, Cluster {int(risk_cluster["cluster"]) + 1} poate fi verificat prioritar,
    deoarece are cel mai mare scor mediu de anomalie. Cluster {int(low_vegetation_cluster["cluster"]) + 1}
    indică zone cu valori medii mai reduse ale indicelui, iar Cluster {int(seasonal_cluster["cluster"]) + 1}
    surprinde cel mai clar variația sezonieră.
    """

    fig_pca = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="cluster_label",
        hover_data={
            "pixel_id": True,
            "window_start": True,
            "window_end": True,
            "risk_score": ":.4f",
            "mean": ":.4f",
            "amplitude": ":.4f",
            "trend_slope": ":.5f",
        },
        title="PCA 3D pe semnături temporale ale pixelilor"
    )

    fig_pca.update_layout(
        height=650,
        legend_title_text="Cluster",
    )

    if len(pca_df) > max_embedding_rows:
        embedding_idx = rng.choice(
            len(pca_df),
            max_embedding_rows,
            replace=False
        )
        embedding_df = pca_df.iloc[embedding_idx].copy()
        X_embedding = X_scaled[embedding_idx]
    else:
        embedding_df = pca_df.copy()
        X_embedding = X_scaled

    tsne_html = ""

    try:
        perplexity = min(
            30,
            max(5, len(X_embedding) // 5)
        )

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )

        X_tsne = tsne.fit_transform(X_embedding)

        tsne_df = embedding_df.copy()
        tsne_df["TSNE1"] = X_tsne[:, 0]
        tsne_df["TSNE2"] = X_tsne[:, 1]

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
            title=f"t-SNE pe eșantion de {len(tsne_df)} ferestre temporale"
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

    except Exception:
        tsne_html = """
        <section class="card reveal active">
            <h2>t-SNE</h2>
            <p class="muted">
                t-SNE nu a putut fi generat pentru configurația curentă.
            </p>
        </section>
        """

    umap_html = ""

    if umap_available:

        try:
            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
            )

            X_umap = reducer.fit_transform(X_embedding)

            umap_df = embedding_df.copy()
            umap_df["UMAP1"] = X_umap[:, 0]
            umap_df["UMAP2"] = X_umap[:, 1]

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
                title=f"UMAP pe eșantion de {len(umap_df)} ferestre temporale"
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

        except Exception:
            umap_html = """
            <section class="card reveal active">
                <h2>UMAP</h2>
                <p class="muted">
                    UMAP nu a putut fi generat pentru configurația curentă.
                </p>
            </section>
            """

    else:
        umap_html = """
        <section class="card reveal active">
            <h2>UMAP</h2>
            <p class="muted">
                UMAP nu este disponibil în mediul curent. Instalează pachetul umap-learn pentru această vizualizare.
            </p>
        </section>
        """

    cluster_profile_fig = go.Figure()
    profile_rows = []

    for cluster_id in range(n_clusters):

        cluster_positions = np.where(map_clusters == cluster_id)[0]

        if len(cluster_positions) == 0:
            continue

        if len(cluster_positions) > max_profile_pixels_per_cluster:
            cluster_positions = rng.choice(
                cluster_positions,
                max_profile_pixels_per_cluster,
                replace=False
            )

        profile_values = []

        for pos in cluster_positions:
            pixel_id = int(valid_indices_all[int(pos)])
            series_values = flat_pixels[pixel_id].astype(float)
            series = make_clean_series(series_values)

            if series.empty:
                continue

            profile_values.append(
                gaussian_filter(
                    series.values,
                    sigma=1
                )
            )

        if len(profile_values) == 0:
            continue

        centroid = np.nanmean(
            np.vstack(profile_values),
            axis=0
        )

        profile_series = pd.Series(
            centroid,
            index=dates
        )

        profile_rows.append(
            (
                f"Cluster {cluster_id + 1}",
                profile_series
            )
        )

        cluster_profile_fig.add_trace(
            go.Scatter(
                x=profile_series.index,
                y=profile_series.values,
                mode="lines",
                name=f"Cluster {cluster_id + 1}",
            )
        )

    cluster_profile_fig.update_layout(
        title=f"Profil temporal mediu pe cluster – {selected_index}",
        xaxis_title="Data",
        yaxis_title=f"{selected_index} [0–1]" if selected_index != "NDMI" else selected_index,
        height=560,
        hovermode="x unified",
        legend_title_text="Cluster",
    )

    fig_cluster_map = px.imshow(
        cluster_grid,
        color_continuous_scale="Turbo",
        aspect="equal",
        title="Hartă clustere pixeli",
        zmin=1,
        zmax=n_clusters,
    )

    fig_cluster_map.update_layout(
        height=650,
        xaxis_title="Coloană pixel",
        yaxis_title="Linie pixel",
        coloraxis_colorbar_title="Cluster",
    )

    fig_risk_map = px.imshow(
        risk_grid,
        color_continuous_scale="Turbo",
        aspect="equal",
        title="Hartă scor anomalie vegetativă"
    )

    fig_risk_map.update_layout(
        height=650,
        xaxis_title="Coloană pixel",
        yaxis_title="Linie pixel",
        coloraxis_colorbar_title="Scor anomalie",
    )

    cluster_centroids = []

    for cluster_label, profile_series in profile_rows:
        cluster_centroids.append({
            "cluster": cluster_label,
            "series": profile_series.reset_index(drop=True)
        })

    if len(cluster_centroids) >= 2:
        dtw_series = [
            (
                item["cluster"],
                item["series"]
            )
            for item in cluster_centroids
        ]

        dtw_matrix = pairwise_dtw_matrix(
            dtw_series
        )

        dtw_labels = [
            item["cluster"]
            for item in cluster_centroids
        ]

        fig_dtw = px.imshow(
            dtw_matrix,
            x=dtw_labels,
            y=dtw_labels,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            title=f"DTW între profilele medii ale clusterelor – {selected_index}"
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
        )

    else:
        dtw_html = """
        <section class="card reveal active">
            <h2>DTW între clustere</h2>
            <p class="muted">
                Nu există suficiente clustere pentru calculul DTW.
            </p>
        </section>
        """

    pixel_std = np.nanstd(pixels_all, axis=1)
    candidate_mask = np.isfinite(pixel_std) & (pixel_std > 1e-5)

    if candidate_mask.any():
        candidate_positions = np.where(candidate_mask)[0]
    else:
        candidate_positions = np.arange(len(pixels_all))

    candidate_risks = map_risk_scores[candidate_positions]

    risk_position = int(
        candidate_positions[np.argmax(candidate_risks)]
    )

    sorted_candidate_positions = candidate_positions[
        np.argsort(candidate_risks)
    ]

    representative_position = int(
        sorted_candidate_positions[len(sorted_candidate_positions) // 2]
    )

    risk_pixel_id = int(valid_indices_all[risk_position])
    representative_pixel_id = int(valid_indices_all[representative_position])

    risk_series = make_clean_series(
        flat_pixels[risk_pixel_id]
    )

    representative_series = make_clean_series(
        flat_pixels[representative_pixel_id]
    )

    risk_series_smooth = pd.Series(
        gaussian_filter(
            risk_series.values.astype(float),
            sigma=1
        ),
        index=dates
    )

    representative_series_smooth = pd.Series(
        gaussian_filter(
            representative_series.values.astype(float),
            sigma=1
        ),
        index=dates
    )

    dtw_pixel_distance = dtw(
        risk_series_smooth.values,
        representative_series_smooth.values
    )

    fig_pixel_compare = go.Figure()

    fig_pixel_compare.add_trace(
        go.Scatter(
            x=risk_series_smooth.index,
            y=risk_series_smooth.values,
            mode="lines",
            name="Pixel de urmărit"
        )
    )

    fig_pixel_compare.add_trace(
        go.Scatter(
            x=representative_series_smooth.index,
            y=representative_series_smooth.values,
            mode="lines",
            name="Pixel reprezentativ"
        )
    )

    fig_pixel_compare.update_layout(
        title=f"Pixel de urmărit vs pixel reprezentativ (DTW={dtw_pixel_distance:.2f})",
        xaxis_title="Data",
        yaxis_title=selected_index,
        height=540,
        showlegend=True,
        hovermode="x unified",
    )

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

    return render_template(
        "base.html",
        title="ML pe pixeli",
        nav_html=render_nav(request.path),
        content=f"""

        <section class="card reveal active">
          <h1>Analiză ML pe pixeli și hartă de risc vegetativ</h1>

          <p class="muted">
            Modulul grupează automat pixelii pe baza evoluției temporale a indicelui spectral selectat.
            Pentru un utilizator practic, precum un fermier sau administrator de teren, rezultatul indică
            zone cu vegetație stabilă, zone cu variație sezonieră și zone care merită verificate în teren.
          </p>

          <form method="get" class="method-box">
            <label><strong>Indice spectral:</strong></label><br>
            <select
                name="index"
                onchange="this.form.submit()"
                class="select-input"
            >
                {index_options}
            </select>

            <br><br>

            <label><strong>ROI:</strong></label><br>
            <select
                name="roi"
                onchange="this.form.submit()"
                class="select-input"
            >
                {roi_options}
            </select>
          </form>

          <div class="method-box">
            <strong>Ce face această analiză?</strong><br><br>
            Aplicația selectează un eșantion de <strong>{max_pixels}</strong> pixeli pentru antrenarea modelului,
            extrage caracteristici temporale din ferestre de <strong>{window}</strong> luni și împarte pixelii în
            <strong>{n_clusters}</strong> clase de comportament. Modelul este apoi aplicat pe toți pixelii validați
            pentru a genera hărți spațiale complete.
          </div>

          <div class="method-box">
            <strong>Pipeline ML:</strong><br><br>
            Pixel spectral
            → eșantionare pentru antrenare
            → mapare pe toți pixelii validați
            → ferestre temporale de {window} luni cu pas {step}
            → feature extraction
            → standardizare
            → K-Means
            → Isolation Forest
            → PCA / t-SNE / UMAP
            → hartă de clustere și hartă de risc.
          </div>
        </section>

        <section class="card reveal active">
            <h2>Rezumat pentru interpretare rapidă</h2>

            <div class="insight-grid reveal active">
                <div class="insight-card">
                    <span class="insight-badge cyan">Cluster dominant</span>
                    <h2>Cluster {int(dominant_cluster["cluster"]) + 1}</h2>
                    <p>
                        Reprezintă cea mai extinsă clasă temporală, cu
                        <strong>{int(dominant_cluster["mapped_pixels"])}</strong> pixeli mapați.
                    </p>
                </div>

                <div class="insight-card">
                    <span class="insight-badge red">Zonă de urmărit</span>
                    <h2>Cluster {int(risk_cluster["cluster"]) + 1}</h2>
                    <p>
                        Are cel mai mare scor mediu de anomalie
                        (<strong>{float(risk_cluster["risk_score"]):.4f}</strong>).
                    </p>
                </div>

                <div class="insight-card">
                    <span class="insight-badge green">Sezonalitate</span>
                    <h2>Cluster {int(seasonal_cluster["cluster"]) + 1}</h2>
                    <p>
                        Are cea mai mare amplitudine medie
                        (<strong>{float(seasonal_cluster["amplitude"]):.4f}</strong>).
                    </p>
                </div>

                <div class="insight-card">
                    <span class="insight-badge gray">Vegetație redusă</span>
                    <h2>Cluster {int(low_vegetation_cluster["cluster"]) + 1}</h2>
                    <p>
                        Are cea mai mică valoare medie a indicelui
                        (<strong>{float(low_vegetation_cluster["mean_value"]):.4f}</strong>).
                    </p>
                </div>
            </div>

            <div class="method-box">
                <strong>Recomandare practică:</strong><br>
                {farmer_recommendation}
            </div>

            <div class="method-box">
                <strong>Calitatea separării:</strong><br>
                {cluster_quality_text}
            </div>
        </section>

        <section class="card reveal active">
            <h2>Rezumat tehnic ML</h2>

            <div class="table-wrap">
                <table class="stats-table">
                    <tbody>
                        <tr>
                            <td>Indice spectral</td>
                            <td>{selected_index}</td>
                        </tr>
                        <tr>
                            <td>ROI</td>
                            <td>{roi.upper()}</td>
                        </tr>
                        <tr>
                            <td>Pixeli validați disponibili</td>
                            <td>{total_valid_pixels}</td>
                        </tr>
                        <tr>
                            <td>Pixeli eșantionați pentru model</td>
                            <td>{len(clean_pixel_series)}</td>
                        </tr>
                        <tr>
                            <td>Pixeli mapați în hărți</td>
                            <td>{len(valid_indices_all)}</td>
                        </tr>
                        <tr>
                            <td>Ferestre temporale extrase</td>
                            <td>{len(features_df)}</td>
                        </tr>
                        <tr>
                            <td>Număr clustere folosite</td>
                            <td>{n_clusters}</td>
                        </tr>
                        <tr>
                            <td>Silhouette Score</td>
                            <td>{metric_silhouette}</td>
                        </tr>
                        <tr>
                            <td>Calinski-Harabasz</td>
                            <td>{metric_calinski}</td>
                        </tr>
                        <tr>
                            <td>Davies-Bouldin</td>
                            <td>{metric_davies}</td>
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
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </section>

        {figure_card(
            fig_cluster_map,
            "Hartă clustere pixeli",
            "Fiecare pixel valid este mapat înapoi pe grila imaginii pe baza comportamentului temporal dominant. Zonele cu aceeași culoare au evoluții temporale asemănătoare.",
            section_id="cluster_map",
        )}

        {figure_card(
            fig_risk_map,
            "Hartă risc/anomalie temporală",
            "Zonele luminoase indică pixeli cu comportament temporal mai atipic conform Isolation Forest. Aceste zone pot fi verificate prioritar.",
            section_id="risk_map",
        )}

        {figure_card(
            cluster_profile_fig,
            "Profil temporal mediu pe cluster",
            "Graficul arată evoluția medie a indicelui pentru fiecare cluster. Este util pentru înțelegerea diferențelor dintre zone.",
            section_id="cluster_profiles",
        )}

        {figure_card(
            fig_pca,
            "PCA 3D",
            "Fiecare punct reprezintă o fereastră temporală extrasă dintr-un pixel. Gruparea punctelor arată similaritatea semnăturilor temporale.",
            section_id="pca_3d",
        )}

        {tsne_html}

        {umap_html}

        {dtw_html}

        {figure_card(
            fig_pixel_compare,
            "Pixel de urmărit vs pixel reprezentativ",
            "Comparație temporală între un pixel cu scor ridicat de anomalie și un pixel reprezentativ al comportamentului mediu.",
            section_id="dtw_pixel"
        )}

        <section class="card reveal active">
            <h2>Notă de interpretare</h2>
            <div class="method-box">
                Analiza este nesupervizată: modelul nu primește etichete reale despre starea terenului.
                Clusterele și scorurile de anomalie indică diferențe statistice în comportamentul temporal,
                iar interpretarea lor trebuie corelată cu observații din teren sau cu informații suplimentare.
            </div>
        </section>

        """
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


