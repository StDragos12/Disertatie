import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw

try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    from google.cloud import storage
except Exception:
    storage = None

from services.indices_service import load_index_array, load_index_dataframe
from utils.ts_utils import extract_features, pairwise_dtw_matrix


DEFAULT_INDICES = ["NDVI", "NDMI", "SAVI", "AVI", "EVI", "GNDVI"]
DEFAULT_ROIS = ["roi1", "roi2"]
DEFAULT_PIXEL_COUNTS = [500, 1000, 2000, 5000]


def round_float(value, digits=6):
    try:
        value = float(value)
        if not np.isfinite(value):
            return None
        return round(value, digits)
    except Exception:
        return None


def matrix_to_list(matrix, digits=6):
    matrix = np.asarray(matrix, dtype=float)
    out = []
    for row in matrix:
        out.append([
            None if not np.isfinite(x) else round(float(x), digits)
            for x in row
        ])
    return out


def get_dates(index_name, roi, series_length):
    try:
        df = load_index_dataframe(index_name)
        df = df[df["roi"].str.lower() == roi.lower()].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        dates = df["date"].dropna().drop_duplicates().sort_values().tolist()
        if len(dates) == series_length:
            return pd.DatetimeIndex(dates)
    except Exception:
        pass

    return pd.date_range(start="2017-01-01", periods=series_length, freq="MS")


def clean_series(values, dates):
    series = pd.Series(np.asarray(values, dtype=float), index=dates)
    series = series.replace([np.inf, -np.inf], np.nan)

    if series.isna().mean() > 0.30:
        return pd.Series(dtype=float)

    series = series.interpolate(method="time").bfill().ffill()

    if series.isna().any():
        return pd.Series(dtype=float)

    return series.astype(float)


def vectorized_features(pixel_matrix):
    matrix = np.asarray(pixel_matrix, dtype=float)
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

    df = pd.DataFrame({
        "mean": row_mean,
        "std": row_std,
        "amplitude": row_max - row_min,
        "trend_slope": trend_slope,
        "anomaly_count": anomaly_count,
    })

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    return df


def upload_json(bucket_name, blob_name, payload):
    if storage is None:
        raise ImportError("google-cloud-storage nu este instalat.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        tmp = fh.name

    blob.upload_from_filename(tmp, content_type="application/json")
    Path(tmp).unlink(missing_ok=True)


def cluster_interpretation(summary, row):
    if int(row["mapped_pixels"]) == 0:
        return "cluster fără pixeli mapați"

    if row["risk_score"] >= summary["risk_score"].quantile(0.75):
        return "zonă de urmărit / comportament atipic"

    if row["amplitude"] >= summary["amplitude"].quantile(0.75):
        return "sezonalitate puternică"

    if row["mean_value"] <= summary["mean_value"].quantile(0.25):
        return "vegetație redusă"

    if abs(row["trend_slope"]) <= 0.002:
        return "comportament stabil"

    return "tendință ascendentă" if row["trend_slope"] > 0 else "tendință descendentă"


def compute_payload(index_name, roi, pixel_count):
    index_name = index_name.upper()
    roi = roi.lower()
    pixel_count = int(pixel_count)

    print(f"[INFO] Loading {index_name}/{roi}")
    arr = load_index_array(index_name, roi)
    height, width, series_length = arr.shape

    flat_pixels = arr.reshape(-1, series_length)
    valid_mask = np.isfinite(flat_pixels).mean(axis=1) >= 0.70
    valid_indices_all = np.where(valid_mask)[0]
    pixels_all = flat_pixels[valid_mask]

    if len(pixels_all) == 0:
        raise ValueError(f"Nu există pixeli validați pentru {index_name}/{roi}")

    dates = get_dates(index_name, roi, series_length)
    rng = np.random.default_rng(42)

    if len(pixels_all) > pixel_count:
        sampled = rng.choice(len(pixels_all), pixel_count, replace=False)
        pixels = pixels_all[sampled]
        valid_indices = valid_indices_all[sampled]
    else:
        pixels = pixels_all
        valid_indices = valid_indices_all

    window = 12
    step = 6
    n_clusters = 4
    feature_rows = []
    clean_count = 0

    for local_id, values in enumerate(pixels):
        pixel_id = int(valid_indices[local_id])
        series = clean_series(values, dates)

        if series.empty or len(series) < window:
            continue

        clean_count += 1

        for start in range(0, len(series) - window + 1, step):
            sub = series.iloc[start:start + window]
            features = extract_features(sub)
            feature_rows.append({
                "pixel_id": pixel_id,
                "window_start": sub.index[0].strftime("%Y-%m"),
                "window_end": sub.index[-1].strftime("%Y-%m"),
                "mean": features["mean"],
                "std": features["std"],
                "amplitude": features["amplitude"],
                "trend_slope": features["trend_slope"],
                "anomaly_count": features["anomaly_count"],
            })

    features_df = pd.DataFrame(feature_rows)
    if features_df.empty or len(features_df) < n_clusters:
        raise ValueError(f"Nu sunt suficiente ferestre pentru {index_name}/{roi}/{pixel_count}")

    feature_cols = ["mean", "std", "amplitude", "trend_slope", "anomaly_count"]
    X = features_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[INFO] Training models")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)

    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    risk_scores = -iso.decision_function(X_scaled)

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = features_df.copy()
    pca_df["cluster"] = clusters.astype(int)
    pca_df["cluster_label"] = [f"Cluster {x + 1}" for x in clusters]
    pca_df["risk_score"] = risk_scores
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]
    pca_df["PC3"] = X_pca[:, 2]

    silhouette = calinski = davies = None
    try:
        metric_n = min(1500, len(X_scaled))
        if len(X_scaled) > metric_n:
            metric_idx = rng.choice(len(X_scaled), metric_n, replace=False)
            X_metric = X_scaled[metric_idx]
            y_metric = clusters[metric_idx]
        else:
            X_metric = X_scaled
            y_metric = clusters

        silhouette = silhouette_score(X_metric, y_metric)
        calinski = calinski_harabasz_score(X_metric, y_metric)
        davies = davies_bouldin_score(X_metric, y_metric)
    except Exception as exc:
        print(f"[WARN] Metrics failed: {exc}")

    print("[INFO] Mapping all pixels")
    map_features = vectorized_features(pixels_all)
    X_map = scaler.transform(map_features[feature_cols].astype(float))
    map_clusters = kmeans.predict(X_map)
    map_risk = -iso.decision_function(X_map)

    cluster_grid = np.full((height, width), np.nan)
    risk_grid = np.full((height, width), np.nan)

    for pos, pixel_id in enumerate(valid_indices_all):
        row = int(pixel_id) // width
        col = int(pixel_id) % width
        cluster_grid[row, col] = int(map_clusters[pos]) + 1
        risk_grid[row, col] = float(map_risk[pos])

    counts = pd.Series(map_clusters).value_counts().reindex(range(n_clusters), fill_value=0).to_dict()

    summary = (
        pca_df
        .groupby("cluster")
        .agg(
            windows=("pixel_id", "count"),
            unique_pixels=("pixel_id", "nunique"),
            mean_value=("mean", "mean"),
            amplitude=("amplitude", "mean"),
            trend_slope=("trend_slope", "mean"),
            risk_score=("risk_score", "mean"),
        )
        .reindex(range(n_clusters))
        .reset_index()
    )

    summary["windows"] = summary["windows"].fillna(0).astype(int)
    summary["unique_pixels"] = summary["unique_pixels"].fillna(0).astype(int)
    for col in ["mean_value", "amplitude", "trend_slope", "risk_score"]:
        summary[col] = summary[col].fillna(0.0)

    summary["mapped_pixels"] = summary["cluster"].map(lambda x: int(counts.get(int(x), 0)))
    summary["interpretation"] = summary.apply(lambda row: cluster_interpretation(summary, row), axis=1)

    print("[INFO] Computing profiles and DTW")
    profiles = []
    profile_rows = []

    for cluster_id in range(n_clusters):
        label = f"Cluster {cluster_id + 1}"

        sample_pixel_ids = (
            pca_df[pca_df["cluster"] == cluster_id]["pixel_id"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
    )

        if len(sample_pixel_ids) == 0:
            positions = np.where(map_clusters == cluster_id)[0]

            if len(positions) > 0:
                if len(positions) > 500:
                    positions = rng.choice(positions, 500, replace=False)

                sample_pixel_ids = [
                int(valid_indices_all[int(pos)])
                for pos in positions
            ]

        if len(sample_pixel_ids) == 0:
            print(f"[WARN] No pixels available for {label}")
            continue

        if len(sample_pixel_ids) > 500:
            sample_pixel_ids = rng.choice(
            sample_pixel_ids,
            500,
            replace=False
        ).tolist()

        values = []

        for pixel_id in sample_pixel_ids:
            series = clean_series(
            flat_pixels[int(pixel_id)],
            dates
        )

            if not series.empty:
                values.append(
                gaussian_filter(
                    series.values.astype(float),
                    sigma=1
                )
            )

        if not values:
            print(f"[WARN] No valid temporal profiles for {label}")
            continue

        centroid = np.nanmean(
        np.vstack(values),
        axis=0
    )

        profiles.append({
        "label": label,
        "dates": [
            d.strftime("%Y-%m-%d")
            for d in dates
        ],
        "values": [
            round_float(x)
            for x in centroid
        ],
    })

        profile_rows.append(
        (
            label,
            pd.Series(centroid).reset_index(drop=True)
        )
    )

    if len(profile_rows) >= 2:
        dtw_matrix = matrix_to_list(pairwise_dtw_matrix(profile_rows), digits=4)
        dtw_labels = [x[0] for x in profile_rows]
    else:
        dtw_matrix = []
        dtw_labels = []

    print("[INFO] Computing embeddings")
    max_embedding = min(1500, max(500, pixel_count))
    if len(pca_df) > max_embedding:
        emb_idx = rng.choice(len(pca_df), max_embedding, replace=False)
        emb_df = pca_df.iloc[emb_idx].copy()
        X_emb = X_scaled[emb_idx]
    else:
        emb_df = pca_df.copy()
        X_emb = X_scaled

    pca_points = []
    for _, row in emb_df.iterrows():
        pca_points.append({
            "PC1": round_float(row["PC1"]),
            "PC2": round_float(row["PC2"]),
            "PC3": round_float(row["PC3"]),
            "cluster_label": row["cluster_label"],
            "pixel_id": int(row["pixel_id"]),
            "risk_score": round_float(row["risk_score"]),
            "mean": round_float(row["mean"]),
            "amplitude": round_float(row["amplitude"]),
            "window_start": row["window_start"],
            "window_end": row["window_end"],
        })

    tsne_points = []
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(X_emb) // 5)), random_state=42, init="pca", learning_rate="auto")
        X_tsne = tsne.fit_transform(X_emb)

        for idx, (_, row) in enumerate(emb_df.iterrows()):
            tsne_points.append({
                "TSNE1": round_float(X_tsne[idx, 0]),
                "TSNE2": round_float(X_tsne[idx, 1]),
                "cluster_label": row["cluster_label"],
                "pixel_id": int(row["pixel_id"]),
                "risk_score": round_float(row["risk_score"]),
                "mean": round_float(row["mean"]),
                "amplitude": round_float(row["amplitude"]),
            })
    except Exception as exc:
        print(f"[WARN] t-SNE failed: {exc}")

    umap_points = []
    if UMAP_AVAILABLE:
        try:
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(X_emb)

            for idx, (_, row) in enumerate(emb_df.iterrows()):
                umap_points.append({
                    "UMAP1": round_float(X_umap[idx, 0]),
                    "UMAP2": round_float(X_umap[idx, 1]),
                    "cluster_label": row["cluster_label"],
                    "pixel_id": int(row["pixel_id"]),
                    "risk_score": round_float(row["risk_score"]),
                    "mean": round_float(row["mean"]),
                    "amplitude": round_float(row["amplitude"]),
                })
        except Exception as exc:
            print(f"[WARN] UMAP failed: {exc}")

    pixel_std = np.nanstd(pixels_all, axis=1)
    candidate_positions = np.where(np.isfinite(pixel_std) & (pixel_std > 1e-5))[0]
    if len(candidate_positions) == 0:
        candidate_positions = np.arange(len(pixels_all))

    candidate_risks = map_risk[candidate_positions]
    risk_position = int(candidate_positions[np.argmax(candidate_risks)])
    sorted_candidates = candidate_positions[np.argsort(candidate_risks)]
    representative_position = int(sorted_candidates[len(sorted_candidates) // 2])

    risk_pixel_id = int(valid_indices_all[risk_position])
    representative_pixel_id = int(valid_indices_all[representative_position])

    risk_series = clean_series(flat_pixels[risk_pixel_id], dates)
    rep_series = clean_series(flat_pixels[representative_pixel_id], dates)

    risk_smooth = gaussian_filter(risk_series.values.astype(float), sigma=1)
    rep_smooth = gaussian_filter(rep_series.values.astype(float), sigma=1)
    pixel_dtw = float(dtw(risk_smooth, rep_smooth))

    dominant = summary.sort_values("mapped_pixels", ascending=False).iloc[0]
    risk_cluster = summary.sort_values("risk_score", ascending=False).iloc[0]
    seasonal = summary.sort_values("amplitude", ascending=False).iloc[0]
    low = summary.sort_values("mean_value", ascending=True).iloc[0]

    payload = {
        "metadata": {
            "index": index_name,
            "roi": roi,
            "pixel_count": int(pixel_count),
            "valid_pixels_total": int(len(pixels_all)),
            "sampled_pixels_used": int(clean_count),
            "mapped_pixels": int(len(valid_indices_all)),
            "windows_extracted": int(len(features_df)),
            "n_clusters": int(n_clusters),
            "height": int(height),
            "width": int(width),
            "window": int(window),
            "step": int(step),
        },
        "metrics": {
            "silhouette": round_float(silhouette, 4),
            "calinski_harabasz": round_float(calinski, 2),
            "davies_bouldin": round_float(davies, 4),
        },
        "highlights": {
            "dominant_cluster": int(dominant["cluster"]) + 1,
            "dominant_pixels": int(dominant["mapped_pixels"]),
            "risk_cluster": int(risk_cluster["cluster"]) + 1,
            "risk_score": round_float(risk_cluster["risk_score"], 4),
            "seasonal_cluster": int(seasonal["cluster"]) + 1,
            "seasonal_amplitude": round_float(seasonal["amplitude"], 4),
            "low_vegetation_cluster": int(low["cluster"]) + 1,
            "low_vegetation_mean": round_float(low["mean_value"], 4),
        },
        "cluster_summary": [
            {
                "cluster": int(row["cluster"]) + 1,
                "sample_pixels": int(row["unique_pixels"]),
                "mapped_pixels": int(row["mapped_pixels"]),
                "windows": int(row["windows"]),
                "mean": round_float(row["mean_value"]),
                "amplitude": round_float(row["amplitude"]),
                "trend": round_float(row["trend_slope"]),
                "risk_score": round_float(row["risk_score"]),
                "interpretation": str(row["interpretation"]),
            }
            for _, row in summary.iterrows()
        ],
        "cluster_grid": matrix_to_list(cluster_grid, digits=0),
        "risk_grid": matrix_to_list(risk_grid, digits=5),
        "cluster_profiles": profiles,
        "dtw": {"labels": dtw_labels, "matrix": dtw_matrix},
        "pca_points": pca_points,
        "tsne_points": tsne_points,
        "umap_points": umap_points,
        "pixel_compare": {
            "risk_pixel_id": risk_pixel_id,
            "representative_pixel_id": representative_pixel_id,
            "dtw": round_float(pixel_dtw, 4),
            "risk": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "values": [round_float(x) for x in risk_smooth],
            },
            "representative": {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "values": [round_float(x) for x in rep_smooth],
            },
        },
    }

    return payload


def precompute_one(bucket_name, index_name, roi, pixel_count):
    payload = compute_payload(index_name, roi, int(pixel_count))
    blob_name = f"data/precomputed_ml/{index_name.upper()}/{roi.lower()}/pixels_{int(pixel_count)}/result.json"
    print(f"[INFO] Uploading gs://{bucket_name}/{blob_name}")
    upload_json(bucket_name, blob_name, payload)
    print(f"[DONE] {index_name}/{roi}/pixels_{pixel_count}")


def parse_csv(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=os.getenv("GCS_BUCKET_NAME"))
    parser.add_argument("--indices", default=",".join(DEFAULT_INDICES))
    parser.add_argument("--rois", default=",".join(DEFAULT_ROIS))
    parser.add_argument("--pixels", default=",".join(str(x) for x in DEFAULT_PIXEL_COUNTS))
    args = parser.parse_args()

    if not args.bucket:
        raise ValueError("Lipsește bucket-ul. Setează --bucket sau GCS_BUCKET_NAME.")

    for index_name in parse_csv(args.indices):
        for roi in parse_csv(args.rois):
            for pixel_count in parse_csv(args.pixels):
                precompute_one(args.bucket, index_name, roi, int(pixel_count))


if __name__ == "__main__":
    main()
