from flask import Blueprint, render_template, request, send_file, redirect, url_for
import io
from services.cloud_run_service import trigger_user_dataset_precompute
from services.dataset_service import (
    DEMO_DATASET_ID,
    list_datasets,
    save_uploaded_dataset,
    get_dataset_csv_bytes,
    delete_dataset,
    inspect_npy_file,
    update_dataset_status,
)
from utils.nav import render_nav


datasets_bp = Blueprint("datasets", __name__)


def _datasets_table() -> str:
    rows = ""

    for dataset in list_datasets(include_demo=True):
        dataset_id = dataset.get("dataset_id", "")
        display_name = dataset.get("display_name", dataset_id)
        source = dataset.get("source", "-")
        status = dataset.get("status", "-")
        input_type = dataset.get("input_type", dataset.get("original_input_type", "-"))
        original_input_type = dataset.get("original_input_type", "-")
        index_detection = dataset.get("index_detection", "-")
        rows_count = dataset.get("rows", "-")
        rois = ", ".join(dataset.get("rois", [])) if dataset.get("rois") else "roi1, roi2" if dataset_id == DEMO_DATASET_ID else "-"
        indices = ", ".join(dataset.get("indices", [])) if dataset.get("indices") else "-"

        delete_html = ""
        if dataset_id != DEMO_DATASET_ID:
            delete_html = f"""
            <form method="post" action="/datasets/{dataset_id}/delete" style="display:inline;"
                  onsubmit="return confirm('Ștergi datasetul {display_name}?');">
                <button class="btn-link danger-link" type="submit">Șterge</button>
            </form>
            """

        rows += f"""
        <tr>
            <td><strong>{display_name}</strong><br><span class="muted">{dataset_id}</span></td>
            <td>{source}<br><span class="muted">input: {original_input_type}</span><br><span class="muted">indice: {index_detection}</span></td>
            <td>{status}</td>
            <td>{input_type}</td>
            <td>{rows_count}</td>
            <td>{rois}</td>
            <td>{indices}</td>
            <td>
                <a class="btn-link secondary" href="/spectral-indices?dataset={dataset_id}">Analiză</a>
                <a class="btn-link secondary" href="/ml-features?dataset={dataset_id}">ML</a>
                <a class="btn-link secondary" href="/datasets/{dataset_id}/download">CSV</a>
                {delete_html}
            </td>
        </tr>
        """

    return rows


def _render_info_rows(info: dict) -> str:
    rows = ""
    for key in [
        "filename", "shape", "ndim", "dtype", "inferred_index", "time_steps",
        "height", "width", "valid_pixels", "finite_values", "nan_values", "min", "max", "mean",
    ]:
        if key in info:
            rows += f"<tr><td>{key}</td><td>{info[key]}</td></tr>"
    return rows


def _npy_preview_html(info: dict | None) -> str:
    if not info:
        return ""

    supported = "Da" if info.get("supported_for_upload") else "Nu"

    if info.get("type") == "zip_npy_collection":
        file_rows = ""
        for item in info.get("files", []):
            file_rows += f"""
            <tr>
                <td>{item.get('filename')}</td>
                <td>{item.get('inferred_index', 'nedetectat')}</td>
                <td>{item.get('shape')}</td>
                <td>{item.get('valid_pixels', '-')}</td>
                <td>{'Da' if item.get('supported_for_upload') else 'Nu'}</td>
            </tr>
            """

        return f"""
        <section class="card reveal active">
            <h2>Rezultat inspectare arhivă NPY</h2>
            <div class="method-box">
                <strong>Compatibil cu upload:</strong> {supported}<br>
                <strong>Fișiere NPY găsite:</strong> {info.get('files_count')}<br>
                <strong>Fișiere compatibile:</strong> {info.get('supported_files')}<br>
                <strong>Indici detectați:</strong> {', '.join(info.get('indices_detected', [])) or 'niciun indice detectat'}<br><br>
                Criteriul de detectare este numele fișierelor din arhivă: <code>NDVI.npy</code>,
                <code>NDMI.npy</code>, <code>SAVI.npy</code>, <code>AVI.npy</code>, <code>EVI.npy</code>,
                <code>GNDVI.npy</code>.
            </div>
            <div class="table-wrap">
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Fișier</th>
                            <th>Indice detectat</th>
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
        "Fișierul poate fi încărcat ca dataset NPY. Va fi convertit automat într-un CSV pixel-level."
        if info.get("supported_for_upload")
        else "Pentru upload NPY este necesar un array numeric 3D cu forma [timp, rânduri, coloane]."
    )

    return f"""
    <section class="card reveal active">
        <h2>Rezultat inspectare NPY</h2>
        <div class="method-box">
            <strong>Compatibil cu upload ML:</strong> {supported}<br>
            <strong>Indice detectat din numele fișierului:</strong> {info.get('inferred_index', 'nedetectat')}<br><br>
            {advice}
        </div>
        <div class="table-wrap">
            <table class="stats-table">
                <tbody>{_render_info_rows(info)}</tbody>
            </table>
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
            if action == "inspect_npy":
                uploaded_file = request.files.get("npy_preview_file")
                info = inspect_npy_file(uploaded_file)
                preview_html = _npy_preview_html(info)
            else:
                display_name = request.form.get("display_name", "Dataset utilizator")
                uploaded_file = request.files.get("file")
                roi_name = request.form.get("roi_name", "parcela1")
                index_name = request.form.get("index_name", "NDVI")
                start_date = request.form.get("start_date", "2021-01")

                record = save_uploaded_dataset(
                    display_name=display_name,
                    file_storage=uploaded_file,
                    roi_name=roi_name,
                    index_name=index_name,
                    start_date=start_date,
                )
                dataset_id = record["dataset_id"]

                trigger_message = ""
                if record.get("input_type") == "pixel_csv":
                    try:
                        operation_name = trigger_user_dataset_precompute(
                            dataset_id=dataset_id,
                            pixel_counts="500,1000,2000,5000",
                        )
                        update_dataset_status(
                            dataset_id=dataset_id,
                            status="processing",
                            message=f"Cloud Run Job pornit automat: {operation_name}",
                        )
                        trigger_message = (
                            "Preprocesarea ML a fost pornită automat în Cloud Run. "
                            "Rezultatele vor apărea în ML Features după finalizarea jobului."
                        )
                    except Exception as trigger_exc:
                        update_dataset_status(
                            dataset_id=dataset_id,
                            status="uploaded",
                            message=(
                                "Datasetul a fost încărcat, dar Cloud Run Job nu a putut fi pornit automat. "
                                f"Detalii: {trigger_exc}"
                            ),
                        )
                        trigger_message = (
                            "Datasetul a fost încărcat, dar jobul Cloud Run nu a pornit automat. "
                            "Poți rula jobul manual din Cloud Shell."
                        )
                else:
                    update_dataset_status(
                        dataset_id=dataset_id,
                        status="completed",
                        message="Dataset ROI-level disponibil pentru analiză temporală, Cross-Index și forecast.",
                    )
                    trigger_message = (
                        "Datasetul este ROI-level, deci nu necesită preprocesare ML pixel-level."
                    )

                message_html = f"""
                <div class="method-box">
                    <strong>Dataset încărcat cu succes:</strong><br>
                    {record["display_name"]} ({dataset_id})<br>
                    Tip: {record.get("input_type", "-")}<br>
                    Status: {trigger_message}<br><br>
                    <a class="btn-link" href="/spectral-indices?dataset={dataset_id}">Deschide analiza</a>
                    <a class="btn-link secondary" href="/ml-features?dataset={dataset_id}">ML Features</a>
                    <a class="btn-link secondary" href="/datasets/{dataset_id}/download">Descarcă CSV generat</a>
                </div>
                """
        except Exception as exc:
            message_html = f"""
            <div class="method-box">
                <strong>Eroare:</strong><br>
                {exc}
            </div>
            """

    rows = _datasets_table()

    return render_template(
        "base.html",
        title="Dataset-uri utilizator",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h1>Dataset-uri utilizator</h1>
            <p class="muted">
                Platforma acceptă CSV ROI-level, CSV pixel-level, NPY 3D singular și arhive ZIP cu mai multe NPY-uri.
                Pentru NPY, fișierele trebuie să aibă forma <strong>[timp, rânduri, coloane]</strong>; aplicația le convertește automat
                într-un CSV pixel-level compatibil cu modulele ML.
            </p>

            <form method="post" enctype="multipart/form-data" class="method-box">
                <input type="hidden" name="action" value="upload">

                <label><strong>Nume dataset:</strong></label><br>
                <input class="select-input" type="text" name="display_name" placeholder="Ex: Ferma Nord 2026" required>

                <br><br>

                <label><strong>Fișier CSV, NPY sau ZIP cu NPY-uri:</strong></label><br>
                <input class="select-input" type="file" name="file" accept=".csv,.npy,.zip" required>

                <br><br>

                <label><strong>Nume parcelă / ROI pentru NPY:</strong></label><br>
                <input class="select-input" type="text" name="roi_name" value="parcela1">
                <p class="muted">Pentru ZIP, același ROI se aplică tuturor fișierelor NPY din arhivă.</p>

                <br><br>

                <label><strong>Indice spectral pentru NPY singular:</strong></label><br>
                <input class="select-input" type="text" name="index_name" value="NDVI">
                <p class="muted">
                    Pentru un singur NPY, aplicația încearcă să deducă indicele din numele fișierului, de exemplu
                    <code>NDMI.npy</code>. Dacă nu îl poate deduce, folosește valoarea introdusă aici.
                    Pentru ZIP, indicele este dedus separat din fiecare fișier din arhivă.
                </p>

                <br><br>

                <label><strong>Prima lună din seria temporală NPY:</strong></label><br>
                <input class="select-input" type="month" name="start_date" value="2021-01">
                <p class="muted">
                    Exemplu: dacă un fișier are forma <code>(36, 24, 32)</code> și prima lună este <code>2021-01</code>,
                    aplicația generează automat observații lunare până în <code>2023-12</code>.
                </p>

                <br><br>

                <button class="btn btn-primary" type="submit">Încarcă dataset</button>
            </form>

            <div class="method-box">
                <strong>CSV ROI-level:</strong><br>
                <code>date,roi,index,value</code><br><br>
                <strong>CSV pixel-level pentru ML:</strong><br>
                <code>date,roi,index,pixel_id,row,col,value</code><br><br>
                <strong>NPY singular pentru ML:</strong><br>
                array numeric 3D <code>[timp, rânduri, coloane]</code>. Dacă fișierul se numește <code>NDMI.npy</code>,
                aplicația îl tratează automat ca NDMI. Dacă numele nu conține un indice recunoscut, se folosește câmpul manual.
                <br><br>
                <strong>ZIP multi-index:</strong><br>
                arhivă cu fișiere denumite <code>NDVI.npy</code>, <code>NDMI.npy</code>, <code>SAVI.npy</code>,
                <code>AVI.npy</code>, <code>EVI.npy</code>, <code>GNDVI.npy</code>. Fiecare fișier devine un indice spectral în același dataset.
            </div>

            {message_html}
        </section>

        <section class="card reveal active">
            <h2>Inspectare NPY înainte de upload</h2>
            <p class="muted">
                Folosește această verificare ca să vezi forma, dimensiunile, numărul de pixeli validați, valorile minime/maxime
                și indicii detectați automat din numele fișierelor.
            </p>
            <form method="post" enctype="multipart/form-data" class="method-box">
                <input type="hidden" name="action" value="inspect_npy">
                <input class="select-input" type="file" name="npy_preview_file" accept=".npy,.zip" required>
                <br><br>
                <button class="btn btn-primary" type="submit">Inspectează NPY / ZIP</button>
            </form>
        </section>

        {preview_html}

        <section class="card reveal active">
            <h2>Dataset-uri disponibile</h2>
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
        """,
    )


@datasets_bp.route("/datasets/<dataset_id>/download")
def download_dataset_csv(dataset_id: str):
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
