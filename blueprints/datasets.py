from __future__ import annotations

import html
import io

from flask import Blueprint, render_template, request, send_file, redirect, url_for

from services.cloud_run_service import trigger_user_dataset_precompute
from services.dataset_service import (
    DEMO_DATASET_ID,
    list_datasets,
    save_uploaded_dataset,
    get_dataset_csv_bytes,
    get_dataset_csv_signed_url,
    delete_dataset,
    inspect_dataset_file,
    has_pixel_level_data,
    update_dataset_status,
    normalize_dataset_id,
    get_dataset_record,
    using_gcs,
)
from utils.nav import render_nav


datasets_bp = Blueprint("datasets", __name__)


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _status_badge(status: str) -> str:
    normalized = str(status or "unknown").strip().lower()

    meta = {
        "completed": {
            "class": "status-completed",
            "icon": "●",
            "label": "Finalizat",
            "hint": "Rezultatele sunt disponibile.",
        },
        "processing": {
            "class": "status-processing",
            "icon": "●",
            "label": "În procesare",
            "hint": "Cloud Run procesează datasetul.",
        },
        "uploaded": {
            "class": "status-uploaded",
            "icon": "●",
            "label": "Încărcat",
            "hint": "Datasetul este încărcat, dar procesarea ML nu este finalizată.",
        },
        "failed": {
            "class": "status-failed",
            "icon": "●",
            "label": "Eșuat",
            "hint": "Procesarea a eșuat. Verifică logurile Cloud Run.",
        },
        "demo": {
            "class": "status-completed",
            "icon": "●",
            "label": "Demo",
            "hint": "Dataset demonstrativ.",
        },
    }

    item = meta.get(
        normalized,
        {
            "class": "status-unknown",
            "icon": "●",
            "label": normalized or "Necunoscut",
            "hint": "Status necunoscut.",
        },
    )

    return f"""
    <span class="dataset-status-pill {item["class"]}" title="{_esc(item["hint"])}" data-status="{_esc(normalized)}">
        <span class="dataset-status-dot">{item["icon"]}</span>
        <span>{_esc(item["label"])}</span>
    </span>
    """


def _first_or_default(values: list[str] | None, default: str) -> str:
    if values:
        return str(values[0])
    return default


def _dataset_action_links(dataset: dict) -> str:
    dataset_id = dataset.get("dataset_id", "")
    rois = dataset.get("rois") or []
    indices = dataset.get("indices") or []

    default_roi = "roi1" if dataset_id == DEMO_DATASET_ID else _first_or_default(rois, "parcela1")
    default_index = _first_or_default(indices, "NDVI")

    analysis_url = f"/spectral-indices?dataset={_esc(dataset_id)}"
    ml_url = (
        f"/ml-features?dataset={_esc(dataset_id)}"
        f"&index={_esc(default_index)}"
        f"&roi={_esc(default_roi)}"
        f"&pixels=500"
    )
    csv_url = f"/datasets/{_esc(dataset_id)}/download"

    delete_html = ""
    if dataset_id != DEMO_DATASET_ID:
        display_name = dataset.get("display_name", dataset_id)
        delete_html = f"""
        <form method="post" action="/datasets/{_esc(dataset_id)}/delete" class="inline-form"
              onsubmit="return confirm('Ștergi datasetul {_esc(display_name)}?');">
            <button class="btn-link danger-link" type="submit">Șterge</button>
        </form>
        """

    return f"""
    <div class="dataset-actions">
        <a class="btn-link secondary" href="{analysis_url}">Analiză</a>
        <a class="btn-link secondary" href="{ml_url}">ML</a>
        <a class="btn-link secondary" href="{csv_url}">CSV</a>
        {delete_html}
    </div>
    """


def _datasets_table() -> tuple[str, bool]:
    rows = ""
    has_processing = False

    for dataset in list_datasets(include_demo=True):
        dataset_id = dataset.get("dataset_id", "")
        display_name = dataset.get("display_name", dataset_id)
        source = dataset.get("source", "-")
        status = dataset.get("status", "completed" if dataset_id == DEMO_DATASET_ID else "-")
        status_message = dataset.get("status_message", dataset.get("message", ""))

        if str(status).lower() == "processing":
            has_processing = True

        input_type = dataset.get("input_type", dataset.get("original_input_type", "-"))
        original_input_type = dataset.get("original_input_type", "-")
        index_detection = dataset.get("index_detection", "-")
        rows_count = dataset.get("rows", "-")

        if dataset.get("rois"):
            rois = ", ".join(dataset.get("rois", []))
        elif dataset_id == DEMO_DATASET_ID:
            rois = "roi1, roi2"
        else:
            rois = "-"

        indices = ", ".join(dataset.get("indices", [])) if dataset.get("indices") else "-"

        status_hint = ""
        if status_message:
            status_hint = f"<br><span class='muted small-muted'>{_esc(status_message)}</span>"

        rows += f"""
        <tr>
            <td>
                <strong>{_esc(display_name)}</strong><br>
                <span class="muted small-muted">{_esc(dataset_id)}</span>
            </td>
            <td>
                {_esc(source)}<br>
                <span class="muted small-muted">input: {_esc(original_input_type)}</span><br>
                <span class="muted small-muted">indice: {_esc(index_detection)}</span>
            </td>
            <td>
                {_status_badge(status)}
                {status_hint}
            </td>
            <td>{_esc(input_type)}</td>
            <td>{_esc(rows_count)}</td>
            <td>{_esc(rois)}</td>
            <td>{_esc(indices)}</td>
            <td>
                {_dataset_action_links(dataset)}
            </td>
        </tr>
        """

    return rows, has_processing


def _render_info_rows(info: dict) -> str:
    rows = ""

    for key in [
        "filename",
        "shape",
        "ndim",
        "dtype",
        "inferred_index",
        "time_steps",
        "height",
        "width",
        "valid_pixels",
        "finite_values",
        "nan_values",
        "min",
        "max",
        "mean",
    ]:
        if key in info:
            rows += f"<tr><td>{_esc(key)}</td><td>{_esc(info[key])}</td></tr>"

    return rows


def _npy_preview_html(info: dict | None) -> str:
    if not info:
        return ""

    supported = "Da" if info.get("supported_for_upload") else "Nu"

    if info.get("type") == "zip_npy_collection":
        file_rows = ""

        for item in info.get("files", []):
            detected = item.get("inferred_band") or item.get("inferred_index", "nedetectat")
            kind = "Bandă" if item.get("inferred_band") else "Indice"
            file_rows += f"""
            <tr>
                <td>{_esc(item.get("zip_path", item.get("filename")))}</td>
                <td>{_esc(kind)}</td>
                <td>{_esc(detected)}</td>
                <td>{_esc(item.get("shape"))}</td>
                <td>{_esc(item.get("valid_pixels", "-"))}</td>
                <td>{'Da' if item.get("supported_for_upload") else 'Nu'}</td>
            </tr>
            """

        bands_text = ", ".join(info.get("bands_detected", [])) or "nicio bandă detectată"
        indices_text = ", ".join(info.get("indices_detected", [])) or "niciun indice detectat"
        input_kind = info.get("input_kind", "-")
        note = info.get("note", "")

        return f"""
        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h2>Rezultat inspectare arhivă NPY</h2>

            <div class="method-box">
                <strong>Compatibil cu upload:</strong> {supported}<br>
                <strong>Tip detectat:</strong> {_esc(input_kind)}<br>
                <strong>Fișiere NPY găsite:</strong> {_esc(info.get("files_count"))}<br>
                <strong>Benzi detectate:</strong> {_esc(bands_text)}<br>
                <strong>Indici detectați:</strong> {_esc(indices_text)}<br><br>
                {_esc(note)}<br><br>
                Pentru benzi se așteaptă: <code>NIR.npy</code>, <code>RED.npy</code>,
                <code>GREEN.npy</code>, <code>BLUE.npy</code>, <code>SWIR.npy</code>.<br>
                Pentru indici se așteaptă: <code>NDVI.npy</code>, <code>NDMI.npy</code>,
                <code>SAVI.npy</code>, <code>AVI.npy</code>, <code>EVI.npy</code>, <code>GNDVI.npy</code>.
            </div>

            <div class="table-wrap">
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Fișier</th>
                            <th>Tip</th>
                            <th>Detectat</th>
                            <th>Shape</th>
                            <th>Pixeli valizi</th>
                            <th>Compatibil</th>
                        </tr>
                    </thead>
                    <tbody>{file_rows}</tbody>
                </table>
            </div>
        </section>
        """

    advice = (
        "Fișierul poate fi încărcat ca dataset NPY cu indice deja calculat."
        if info.get("supported_for_upload")
        else info.get("note") or "Pentru upload NPY este necesar un array numeric 3D cu forma [timp, rânduri, coloane]."
    )

    return f"""
    <section class="card reveal active">
        <div class="card-top-line"></div>
        <h2>Rezultat inspectare NPY</h2>

        <div class="method-box">
            <strong>Compatibil cu upload:</strong> {supported}<br>
            <strong>Indice detectat:</strong> {_esc(info.get("inferred_index", "nedetectat"))}<br>
            <strong>Bandă detectată:</strong> {_esc(info.get("inferred_band", "-"))}<br><br>
            {_esc(advice)}
        </div>

        <div class="table-wrap">
            <table class="stats-table">
                <tbody>{_render_info_rows(info)}</tbody>
            </table>
        </div>
    </section>
    """



def _csv_preview_html(info: dict | None) -> str:
    if not info:
        return ""

    supported = "Da" if info.get("supported_for_upload") else "Nu"
    input_kind = info.get("input_kind", "-")
    columns = ", ".join(info.get("columns", [])) or "-"
    rois = ", ".join(info.get("rois", [])) or "-"
    indices = ", ".join(info.get("indices_detected", [])) or "-"
    bands = ", ".join(info.get("bands_detected", [])) or "-"
    pixel_level = "Da" if info.get("pixel_level") else "Nu"
    note = info.get("note", "")

    return f"""
    <section class="card reveal active">
        <div class="card-top-line"></div>
        <h2>Rezultat inspectare CSV</h2>

        <div class="method-box">
            <strong>Compatibil cu upload:</strong> {supported}<br>
            <strong>Tip detectat:</strong> {_esc(input_kind)}<br>
            <strong>Rânduri valide:</strong> {_esc(info.get("rows", "-"))}<br>
            <strong>Pixel-level:</strong> {pixel_level}<br>
            <strong>Interval temporal:</strong> {_esc(info.get("date_min", "-"))} – {_esc(info.get("date_max", "-"))}<br>
            <strong>ROI-uri:</strong> {_esc(rois)}<br>
            <strong>Benzi detectate:</strong> {_esc(bands)}<br>
            <strong>Indici detectați / calculați:</strong> {_esc(indices)}<br><br>
            {_esc(note)}
        </div>

        <div class="table-wrap">
            <table class="stats-table">
                <tbody>
                    <tr><td>Fișier</td><td>{_esc(info.get("filename", "-"))}</td></tr>
                    <tr><td>Coloane</td><td>{_esc(columns)}</td></tr>
                </tbody>
            </table>
        </div>
    </section>
    """

def _upload_message(record: dict, cloud_run_started: bool, operation_name: str | None = None) -> str:
    dataset_id = record["dataset_id"]
    input_type = record.get("input_type", "-")

    if cloud_run_started:
        status_html = """
        <span class="dataset-status-pill status-processing">
            <span class="dataset-status-dot">●</span>
            <span>În procesare</span>
        </span>
        """
        note = """
        Procesarea ML a fost pornită automat în Cloud Run. Pagina se va actualiza periodic,
        iar rezultatele vor fi disponibile după generarea fișierelor <code>result.json</code>.
        """
    else:
        status_html = """
        <span class="dataset-status-pill status-completed">
            <span class="dataset-status-dot">●</span>
            <span>Disponibil</span>
        </span>
        """
        note = """
        Datasetul este disponibil pentru analiză temporală, Cross-Index și forecast.
        Pentru ML pe pixeli este necesar CSV pixel-level sau NPY/ZIP 3D.
        """

    operation_html = ""
    if operation_name:
        operation_html = f"""
        <br><span class="muted small-muted">Cloud Run operation: {_esc(operation_name)}</span>
        """

    return f"""
    <div class="method-box dataset-upload-result">
        <div class="dataset-result-header">
            <div>
                <strong>Dataset încărcat cu succes:</strong><br>
                {_esc(record.get("display_name", dataset_id))} ({_esc(dataset_id)})
            </div>
            {status_html}
        </div>

        <p class="muted">
            Tip: <strong>{_esc(input_type)}</strong><br>
            {note}
            {operation_html}
        </p>

        <div class="dataset-actions">
            <a class="btn-link" href="/spectral-indices?dataset={_esc(dataset_id)}">Deschide analiza</a>
            <a class="btn-link secondary" href="/ml-features?dataset={_esc(dataset_id)}">ML Features</a>
            <a class="btn-link secondary" href="/datasets/{_esc(dataset_id)}/download">Descarcă CSV generat</a>
        </div>
    </div>
    """


def _upload_form_html() -> str:
    return """
    <section class="card reveal active dataset-upload-card">
        <div class="card-top-line"></div>

        <div class="section-heading">
            <div>
                <h1>Dataset-uri utilizator</h1>
                <p class="muted">
                    Încarcă CSV cu indici, CSV cu benzi spectrale sau ZIP/NPY cu date 3D.
                </p>
            </div>
        </div>

        <form method="post" enctype="multipart/form-data" class="dataset-upload-form">
            <input type="hidden" name="action" value="upload">

            <div class="dataset-single-form">

                <div class="form-field dataset-row-field">
                    <label for="display_name">Nume dataset</label>
                    <input
                        id="display_name"
                        name="display_name"
                        type="text"
                        placeholder="Ex: Ferma Nord 2026"
                        required
                    >
                </div>

                <div class="form-field dataset-row-field">
                    <label for="file">Fișier CSV, NPY sau ZIP</label>
                    <input
                        id="file"
                        name="file"
                        type="file"
                        accept=".csv,.npy,.zip"
                        required
                    >
                </div>

                <div class="form-field dataset-row-field">
                    <label for="roi_name">ROI implicit</label>
                    <input
                        id="roi_name"
                        name="roi_name"
                        type="text"
                        value="parcela1"
                        required
                    >
                </div>

                <div class="form-field dataset-row-field">
                    <label for="start_date">Start temporal</label>
                    <input
                        id="start_date"
                        name="start_date"
                        type="month"
                        value="2021-01"
                    >
                </div>

                <div class="dataset-format-note">
                    Formate acceptate: CSV cu <code>date,roi,index,value</code>, CSV cu benzi <code>nir,red,green,blue,swir</code>, sau ZIP NPY cu <code>NIR/RED/GREEN/BLUE/SWIR</code> ori <code>NDVI/NDMI/SAVI/AVI/EVI/GNDVI</code>.
                </div>

                <div class="dataset-submit-row">
                    <button type="submit" class="btn-primary">
                        Încarcă dataset
                    </button>
                </div>

            </div>
        </form>
    </section>
    """


def _inspect_form_html() -> str:
    return """
    <section class="card reveal active compact-card dataset-inspect-card">
        <div class="card-top-line"></div>

        <div class="section-heading inspect-heading">
            <div>
                <h2>Inspectare CSV / NPY / ZIP</h2>
                <p class="muted">
                    Verifică rapid tipul fișierului, coloanele detectate, shape-ul NPY și compatibilitatea cu platforma.
                </p>
            </div>
        </div>

        <form method="post" enctype="multipart/form-data" class="inspect-form inspect-inline-form">
            <input type="hidden" name="action" value="inspect_file">

            <div class="inspect-file-field">
                <label for="preview_file">Fișier CSV, NPY sau ZIP</label>
                <input
                    id="preview_file"
                    name="preview_file"
                    type="file"
                    accept=".csv,.npy,.zip"
                    required
                >
            </div>

            <button type="submit" class="btn-link secondary inspect-submit-btn">
                Inspectează fișierul
            </button>
        </form>
    </section>
    """


def _dataset_formats_help_html() -> str:
    return """
    <section class="card reveal active compact-card">
        <div class="card-top-line"></div>
        <h2>Formate acceptate</h2>

        <div class="dataset-format-grid">
            <div class="format-box">
                <h3>CSV cu indici</h3>
                <code>date,roi,index,value</code><br>
                <code>date,roi,index,pixel_id,row,col,value</code>
                <p class="muted">
                    Valorile indicilor sunt deja calculate. Platforma le folosește direct.
                </p>
            </div>

            <div class="format-box">
                <h3>CSV cu benzi</h3>
                <code>date,roi,pixel_id,row,col,nir,red,green,blue,swir</code>
                <p class="muted">
                    Platforma calculează automat NDVI, NDMI, SAVI, EVI, GNDVI și AVI.
                </p>
            </div>

            <div class="format-box">
                <h3>ZIP NPY cu benzi</h3>
                <code>parcela1/NIR.npy</code>, <code>RED.npy</code>, <code>GREEN.npy</code>,
                <code>BLUE.npy</code>, <code>SWIR.npy</code>
                <p class="muted">
                    Fiecare bandă trebuie să fie array 3D: <code>[timp, rânduri, coloane]</code>.
                </p>
            </div>

            <div class="format-box">
                <h3>ZIP/NPY cu indici</h3>
                <code>NDVI.npy</code>, <code>NDMI.npy</code>, <code>SAVI.npy</code>,
                <code>AVI.npy</code>, <code>EVI.npy</code>, <code>GNDVI.npy</code>
                <p class="muted">
                    Valorile sunt citite direct din array. Pixelul este mapat prin row/col.
                </p>
            </div>
        </div>
    </section>
    """


@datasets_bp.route("/datasets", methods=["GET", "POST"])
def datasets_page():
    message_html = ""
    preview_html = ""

    if request.method == "POST":
        action = request.form.get("action", "upload")

        try:
            if action in {"inspect_dataset", "inspect_npy"}:
                uploaded_file = (
                    request.files.get("dataset_preview_file")
                    or request.files.get("npy_preview_file")
                )
                info = inspect_dataset_file(uploaded_file)

                if info.get("type") == "csv_file":
                    preview_html = _csv_preview_html(info)
                else:
                    preview_html = _npy_preview_html(info)

            else:
                display_name = request.form.get("display_name", "Dataset utilizator")
                uploaded_file = request.files.get("file")
                roi_name = request.form.get("roi_name", "parcela1")
                start_date = request.form.get("start_date", "2021-01")

                candidate_dataset_id = normalize_dataset_id(display_name)
                existing_record = get_dataset_record(candidate_dataset_id)

                if existing_record is None:
                    record = save_uploaded_dataset(
                        display_name=display_name,
                        file_storage=uploaded_file,
                        roi_name=roi_name,
                        start_date=start_date,
                    )

                    dataset_id = record["dataset_id"]
                    cloud_run_started = False
                    operation_name = None

                    if has_pixel_level_data(dataset_id):
                        try:
                            operation_name = trigger_user_dataset_precompute(
                                dataset_id=dataset_id,
                                pixel_counts="500,1000,2000,5000",
                            )

                            update_dataset_status(
                                dataset_id=dataset_id,
                                status="processing",
                                message="Procesarea ML a fost pornită automat în Cloud Run.",
                                extra={
                                    "cloud_run_operation": operation_name,
                                },
                            )

                            record["status"] = "processing"
                            record["status_message"] = "Procesarea ML a fost pornită automat în Cloud Run."
                            cloud_run_started = True

                        except Exception as exc:
                            update_dataset_status(
                                dataset_id=dataset_id,
                                status="uploaded",
                                message=f"Datasetul a fost încărcat, dar Cloud Run nu a putut fi pornit automat: {exc}",
                            )

                            record["status"] = "uploaded"
                            record["status_message"] = f"Cloud Run nu a putut fi pornit automat: {exc}"

                    else:
                        update_dataset_status(
                            dataset_id=dataset_id,
                            status="completed",
                            message="Dataset ROI-level disponibil pentru analiză temporală, Cross-Index și forecast.",
                        )
                        record["status"] = "completed"
                        record["status_message"] = "Dataset ROI-level disponibil."

                    message_html = _upload_message(
                        record=record,
                        cloud_run_started=cloud_run_started,
                        operation_name=operation_name,
                    )

                else:
                    message_html = ""

        except Exception as exc:
            message_html = f"""
            <div class="method-box error-box">
                <strong>Eroare:</strong><br>
                {_esc(exc)}
            </div>
            """

    rows, has_processing = _datasets_table()

    auto_refresh_html = ""
    if has_processing:
        auto_refresh_html = """
        <script>
            setTimeout(function () {
                window.location.reload();
            }, 15000);
        </script>
        """

    return render_template(
        "base.html",
        title="Dataset-uri utilizator",
        nav_html=render_nav(request.path),
        content=f"""
        {message_html}

        {_upload_form_html()}

        {_inspect_form_html()}

        {preview_html}

        {_dataset_formats_help_html()}

        <section class="card reveal active">
            <div class="card-top-line"></div>
            <div class="section-heading horizontal-heading">
                <div>
                    <h2>Dataset-uri disponibile</h2>
                    <p class="muted">
                        Statusul indică dacă procesarea Cloud Run este finalizată.
                        Pagina se reîncarcă automat cât timp există dataseturi în procesare.
                    </p>
                </div>
            </div>

            <div class="table-wrap">
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Dataset</th>
                            <th>Sursă</th>
                            <th>Status</th>
                            <th>Tip</th>
                            <th>Rânduri</th>
                            <th>ROI-uri</th>
                            <th>Indici</th>
                            <th>Acțiuni</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
            </div>
        </section>

        {auto_refresh_html}
        """,
    )


@datasets_bp.route("/datasets/<dataset_id>/download")
def download_dataset_csv(dataset_id: str):
    dataset_id = normalize_dataset_id(dataset_id)

    if using_gcs() and dataset_id != DEMO_DATASET_ID:
        signed_url, _ = get_dataset_csv_signed_url(dataset_id)
        return redirect(signed_url)

    data, filename = get_dataset_csv_bytes(dataset_id)

    return send_file(
        io.BytesIO(data),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@datasets_bp.route("/datasets/<dataset_id>/delete", methods=["POST"])
def delete_dataset_route(dataset_id: str):
    delete_dataset(dataset_id)
    return redirect(url_for("datasets.datasets_page"))