from flask import Blueprint, render_template, request
from config import HOME_SECTIONS, ROI_INFO, SERIES_CATALOG
from services.ndvi_service import load_ndvi, get_sites, pretty_site_name
from utils.nav import render_nav

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
def series_catalog_page():
    cards = []

    for group in SERIES_CATALOG:
        items_html = ""
        for item in group["items"]:
            items_html += f"""
            <a class="dataset-link-card" href="{item['href']}">
              <div class="dataset-link-top">
                <strong>{item['label']}</strong>
                <span class="mini-badge">{item['tag']}</span>
              </div>
            </a>
            """

        cards.append(
            f"""
            <div class="card">
              <h2>{group['group']}</h2>
              <p class="muted">{group['description']}</p>
              <div class="dataset-grid">
                {items_html}
              </div>
            </div>
            """
        )

    return render_template(
        "base.html",
        title="Catalog serii",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card">
          <h1>Catalogul seriilor temporale</h1>
          <p class="muted">
            Aplicația este organizată în jurul a trei categorii de serii temporale:
            <strong>serii sintetice</strong>, <strong>serii climatice demonstrative</strong>
            și <strong>serii NDVI Sentinel-2</strong>.
          </p>
        </section>
        {''.join(cards)}
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
        <section class="card">
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
        <section class="card">
          <h1>Metodologie</h1>
          <p class="muted">
            Aplicația NDVI Viz este concepută ca o platformă educațională și analitică pentru
            studiul seriilor temporale, folosind date satelitare NDVI ca studiu principal de caz,
            completate de serii sintetice și serii climatice demonstrative.
          </p>

          <div class="method-box">
            <strong>Scop metodologic:</strong><br>
            Evidențierea comparativă a proprietăților seriilor temporale:
            staționaritate, trend, sezonalitate, anomalii și comportament de forecast.
          </div>

          <div class="method-box">
            <strong>Sursa principală de date:</strong><br>
            Date Sentinel-2, procesate pentru obținerea indicelui NDVI, calculat pe baza benzilor
            <code>B8</code> și <code>B4</code>.
          </div>

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
            <strong>Pipeline analitic:</strong><br>
            1. Încărcarea sau generarea seriei.<br>
            2. Resampling / interpolare la frecvență lunară unde este cazul.<br>
            3. Analiză descriptivă.<br>
            4. Testare staționaritate prin ADF.<br>
            5. Descompunere STL în trend, sezonalitate și reziduu.<br>
            6. Detectarea anomaliilor pe baza reziduurilor.<br>
            7. Forecast prin modele ARIMA/SARIMA și comparație cu LSTM pentru seriile NDVI.
          </div>

          <div class="method-box">
            <strong>Poziționare pentru disertație:</strong><br>
            NDVI rămâne studiul principal de caz, însă aplicația demonstrează că aceleași tehnici
            pot fi aplicate și altor categorii de serii temporale, ceea ce crește valoarea metodologică a proiectului.
          </div>
        </section>
        """,
    )