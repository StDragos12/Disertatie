from flask import Blueprint, render_template, request
import plotly.graph_objects as go

from services.synthetic_service import (
    SYNTHETIC_SERIES_META,
    generate_synthetic_series,
    generate_temperature_demo_series,
)
from utils.insights import generate_insights
from utils.nav import render_nav
from utils.page import figure_card
from utils.ts_utils import (
    stationarity_metrics_from_series,
    count_anomalies_in_series,
    stl_series,
)

synthetic_bp = Blueprint("synthetic", __name__)


THEORY_BY_KIND = {
    "white-noise": {
        "badge": "Staționară",
        "summary": (
            "White noise este o serie de referință fără trend și fără sezonalitate. "
            "Valorile oscilează aleator în jurul unei medii aproximativ constante."
        ),
        "use": (
            "Este utilă pentru verificarea testelor de staționaritate și pentru a arăta "
            "cum se comportă o serie fără structură temporală persistentă."
        ),
    },
    "random-walk": {
        "badge": "Nestaționară",
        "summary": (
            "Random walk este o serie nestaționară în care valoarea curentă depinde de "
            "valoarea anterioară și de o variație aleatoare."
        ),
        "use": (
            "Este folosită pentru a ilustra acumularea variațiilor în timp și dificultatea "
            "modelării unei serii care nu revine la o medie stabilă."
        ),
    },
    "linear-trend": {
        "badge": "Trend",
        "summary": (
            "Seria cu trend liniar conține o direcție clară de evoluție, peste care este "
            "adăugat zgomot aleator."
        ),
        "use": (
            "Este utilă pentru validarea metodelor care identifică trendul și separă "
            "componenta sistematică de variațiile locale."
        ),
    },
    "seasonal-noise": {
        "badge": "Sezonalitate",
        "summary": (
            "Seria sinusoidală cu zgomot conține un tipar periodic repetitiv, asemănător "
            "unui ciclu anual sau sezonier."
        ),
        "use": (
            "Este utilă pentru verificarea descompunerii STL și pentru explicarea "
            "componentelor sezoniere."
        ),
    },
    "trend-seasonal": {
        "badge": "Trend + sezonalitate",
        "summary": (
            "Această serie combină o componentă de trend cu o componentă sezonieră, fiind "
            "mai apropiată de seriile reale."
        ),
        "use": (
            "Este folosită pentru a demonstra că aplicația poate separa simultan evoluția "
            "pe termen lung și variațiile periodice."
        ),
    },
    "temperature_demo": {
        "badge": "Climatică",
        "summary": (
            "Seria demonstrativă de temperatură prezintă un comportament sezonier clar, "
            "determinat de variația anuală a temperaturii."
        ),
        "use": (
            "Este inclusă pentru a arăta că metodologia nu este limitată la NDVI, ci poate "
            "fi aplicată și altor serii temporale."
        ),
    },
}


def safe_round(value, digits=4):
    if value in [None, "N/A"]:
        return "N/A"

    try:
        numeric_value = float(value)

        if abs(numeric_value) < 10 ** (-digits):
            return f"< {10 ** (-digits):.{digits}f}"

        return round(numeric_value, digits)

    except Exception:
        return "N/A"


def build_insights_html(insights):
    if not insights:
        return """
        <p class="muted">
            Nu au fost generate observații automate pentru această serie.
        </p>
        """

    items = ""

    for insight in insights:
        items += f"<li>{insight}</li>"

    return f"""
    <ul class="insights-list">
        {items}
    </ul>
    """


def build_metric_cards(
    observation_count,
    adf_stat,
    p_value,
    stationary,
    anomaly_count,
):
    status_class = "good" if stationary == "Staționară" else "warn"

    return f"""
    <style>
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-top: 18px;
        }}

        .kpi-card {{
            background: rgba(15, 30, 55, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            padding: 18px 20px;
            min-height: 135px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.16);
        }}

        .kpi-label {{
            display: inline-flex;
            align-items: center;
            padding: 5px 10px;
            border-radius: 999px;
            background: rgba(96, 165, 250, 0.15);
            color: #bfdbfe;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 12px;
        }}

        .kpi-value {{
            font-size: 2.1rem;
            line-height: 1.05;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 8px;
        }}

        .kpi-desc {{
            color: #cbd5e1;
            font-size: 0.94rem;
            line-height: 1.45;
        }}

        .kpi-card.good {{
            border-color: rgba(34, 197, 94, 0.35);
        }}

        .kpi-card.good .kpi-label {{
            background: rgba(34, 197, 94, 0.16);
            color: #bbf7d0;
        }}

        .kpi-card.warn {{
            border-color: rgba(251, 191, 36, 0.35);
        }}

        .kpi-card.warn .kpi-label {{
            background: rgba(251, 191, 36, 0.16);
            color: #fde68a;
        }}

        .kpi-card.danger {{
            border-color: rgba(248, 113, 113, 0.35);
        }}

        .kpi-card.danger .kpi-label {{
            background: rgba(248, 113, 113, 0.16);
            color: #fecaca;
        }}
    </style>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Date</div>
            <div class="kpi-value">{observation_count}</div>
            <div class="kpi-desc">observații lunare disponibile pentru analiza seriei.</div>
        </div>

        <div class="kpi-card">
            <div class="kpi-label">ADF statistic</div>
            <div class="kpi-value">{adf_stat}</div>
            <div class="kpi-desc">valoarea statisticii testului Augmented Dickey-Fuller.</div>
        </div>

        <div class="kpi-card">
            <div class="kpi-label">p-value</div>
            <div class="kpi-value">{p_value}</div>
            <div class="kpi-desc">valoare utilizată pentru interpretarea staționarității.</div>
        </div>

        <div class="kpi-card {status_class}">
            <div class="kpi-label">Status</div>
            <div class="kpi-value">{stationary}</div>
            <div class="kpi-desc">interpretare automată a comportamentului seriei.</div>
        </div>

        <div class="kpi-card danger">
            <div class="kpi-label">Anomalii</div>
            <div class="kpi-value">{anomaly_count}</div>
            <div class="kpi-desc">puncte neobișnuite identificate pe baza componentei reziduale.</div>
        </div>
    </div>
    """

def build_theory_section(series_key, title):
    theory = THEORY_BY_KIND.get(
        series_key,
        {
            "badge": "Serie temporală",
            "summary": (
                "Seria este utilizată pentru explorarea comportamentului temporal și "
                "pentru verificarea metodelor de analiză."
            ),
            "use": (
                "Această serie ajută la observarea diferențelor dintre trend, "
                "sezonalitate, staționaritate și variații locale."
            ),
        },
    )

    return f"""
    <section class="card reveal active">
        <h2>Explicație teoretică</h2>

        <div class="method-box">
            <strong>Tip serie:</strong>
            <span class="badge">{theory["badge"]}</span>
            <br><br>
            <strong>Descriere:</strong><br>
            {theory["summary"]}
            <br><br>
            <strong>Rol în aplicație:</strong><br>
            {theory["use"]}
        </div>

        <p class="muted">
            Această secțiune este inclusă pentru a conecta partea vizuală cu interpretarea
            statistică. În acest mod, seria <strong>{title}</strong> nu este doar afișată grafic,
            ci este folosită ca exemplu metodologic pentru analiza seriilor temporale.
        </p>
    </section>
    """


def build_stl_section(series, title, value_label, section_id):
    try:
        result = stl_series(
            series,
            period=12
        )

        fig_components = go.Figure()

        fig_components.add_trace(
            go.Scatter(
                x=result.trend.index,
                y=result.trend.values,
                mode="lines",
                name="Trend",
            )
        )

        fig_components.add_trace(
            go.Scatter(
                x=result.seasonal.index,
                y=result.seasonal.values,
                mode="lines",
                name="Sezonalitate",
            )
        )

        fig_components.add_trace(
            go.Scatter(
                x=result.resid.index,
                y=result.resid.values,
                mode="lines",
                name="Reziduu",
            )
        )

        fig_components.update_layout(
            hovermode="x unified",
            height=560,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=60, r=40, t=90, b=60),
        )

        return figure_card(
            fig_components,
            f"Componente STL – {title}",
            (
                "Descompunerea STL separă seria în trend, sezonalitate și reziduu. "
                "Trendul arată direcția generală, sezonalitatea evidențiază tiparul periodic, "
                "iar reziduul surprinde variațiile rămase."
            ),
            section_id=f"{section_id}_components",
            yaxis_title=value_label,
        )

    except Exception:
        return """
        <section class="card reveal active">
            <h2>Componente STL</h2>
            <p class="muted">
                Descompunerea STL nu este disponibilă pentru această serie. De obicei,
                acest lucru apare când seria este prea scurtă sau nu are suficiente observații
                pentru o sezonalitate anuală stabilă.
            </p>
        </section>
        """


def generic_series_page(
    title: str,
    df_series,
    current_path: str,
    value_label: str = "Valoare",
    series_kind: str = "generic",
    theory_key: str = "generic",
):
    series = (
        df_series
        .set_index("date")["value"]
        .asfreq("MS")
    )

    station = stationarity_metrics_from_series(series)

    anomaly_count = count_anomalies_in_series(
        series,
        period=12
    )

    insights = generate_insights(series)
    insights_html = build_insights_html(insights)

    adf_stat = safe_round(
        station.get("adf_stat", station.get("statistic", "N/A")),
        4,
    )

    p_value = safe_round(
        station.get("p_value", "N/A"),
        4,
    )

    stationary = station.get(
        "stationary",
        "N/A"
    )

    category = df_series["category"].iloc[0]
    description = df_series["description"].iloc[0]
    observation_count = len(df_series)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_series["date"],
            y=df_series["value"],
            mode="lines+markers",
            name=df_series["series_name"].iloc[0],
        )
    )

    fig.update_layout(
        hovermode="x unified",
        height=560,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60, r=40, t=90, b=60),
    )

    metric_cards = build_metric_cards(
        observation_count=observation_count,
        adf_stat=adf_stat,
        p_value=p_value,
        stationary=stationary,
        anomaly_count=anomaly_count,
    )

    theory_section = build_theory_section(
        theory_key,
        title,
    )

    stl_section = build_stl_section(
        series=series,
        title=title,
        value_label=value_label,
        section_id=series_kind,
    )

    content = f"""
    <section class="card reveal active">
        <div class="card-top-line"></div>
        <h1>{title}</h1>

        <p class="muted">
            {description}
        </p>

        <div class="method-box">
            <strong>Categorie:</strong> {category}<br>
            <strong>Tip analiză:</strong> serie temporală lunară<br>
            <strong>Rol:</strong> validarea și explicarea metodelor de analiză temporală.
        </div>
    </section>

    <section class="card reveal active">
        <h2>Rezumat statistic</h2>
        <p class="muted">
            Indicatorii de mai jos sintetizează comportamentul seriei și ajută la interpretarea rapidă
            a staționarității, a numărului de observații și a punctelor neobișnuite detectate.
        </p>
        {metric_cards}
    </section>

    {figure_card(
        fig,
        f"Serie temporală – {title}",
        (
            "Graficul prezintă evoluția lunară a seriei. Forma curbei permite observarea "
            "trendului, sezonalității, variațiilor locale și eventualelor puncte atipice."
        ),
        section_id=f"{series_kind}_main",
        yaxis_title=value_label,
    )}

    <section class="card reveal active">
        <h2>Interpretare automată</h2>

        <p class="muted">
            Observațiile de mai jos sunt generate automat pe baza caracteristicilor statistice
            ale seriei temporale.
        </p>

        {insights_html}
    </section>

    {theory_section}

    {stl_section}

    <section class="card reveal active">
        <h2>Legătura cu analiza NDVI</h2>

        <div class="method-box">
            Seriile sintetice și demonstrative oferă exemple controlate pentru proprietăți
            precum staționaritatea, trendul și sezonalitatea. După validarea acestor concepte,
            aceleași metode pot fi aplicate seriilor reale provenite din indici spectrali,
            precum NDVI, NDMI, SAVI sau EVI.
        </div>
    </section>
    """

    return render_template(
        "base.html",
        title=title,
        nav_html=render_nav(current_path),
        content=content,
    )


@synthetic_bp.route("/synthetic")
def synthetic_index():
    cards = ""

    for key, meta in SYNTHETIC_SERIES_META.items():
        theory = THEORY_BY_KIND.get(
            key,
            {
                "badge": meta["category"],
                "summary": meta["description"],
            },
        )

        cards += f"""
        <div class="catalog-card">
            <div class="badge">{theory["badge"]}</div>
            <h2>{meta["title"]}</h2>

            <p class="muted">
                {meta["description"]}
            </p>

            <div class="method-box compact-box">
                {theory["summary"]}
            </div>

            <ul>
                <li><a href="/synthetic/{key}">Deschide analiza</a></li>
            </ul>
        </div>
        """

    return render_template(
        "base.html",
        title="Serii sintetice",
        nav_html=render_nav(request.path),
        content=f"""
        <section class="card reveal active">
            <div class="card-top-line"></div>
            <h1>Serii sintetice și demonstrative</h1>

            <p class="muted">
                Aceste serii sunt folosite pentru explicarea și validarea conceptelor de bază
                din analiza seriilor temporale: staționaritate, trend, sezonalitate, zgomot,
                decompoziție STL și identificarea anomaliilor.
            </p>

            <div class="method-box">
                <strong>Rol metodologic:</strong><br>
                Înainte de aplicarea metodelor pe date satelitare reale, seriile sintetice oferă
                cazuri controlate, unde comportamentul așteptat este cunoscut. Astfel, utilizatorul
                poate înțelege mai clar ce urmăresc testele statistice și graficele generate.
            </div>
        </section>

        <div class="catalog-grid">
            {cards}
        </div>

        <section class="card reveal active">
            <h2>Utilizare în cadrul aplicației</h2>

            <div class="insight-grid reveal active">
                <div class="insight-card">
                    <span class="insight-badge green">Staționaritate</span>
                    <h2>White Noise</h2>
                    <p>Exemplu de serie fără trend și fără sezonalitate persistentă.</p>
                </div>

                <div class="insight-card">
                    <span class="insight-badge red">Nestaționaritate</span>
                    <h2>Random Walk</h2>
                    <p>Exemplu de serie cu acumulare a variațiilor în timp.</p>
                </div>

                <div class="insight-card">
                    <span class="insight-badge blue">Sezonalitate</span>
                    <h2>Sinusoidală</h2>
                    <p>Exemplu de serie cu tipar periodic clar.</p>
                </div>

                <div class="insight-card">
                    <span class="insight-badge purple">Caz mixt</span>
                    <h2>Trend + sezon</h2>
                    <p>Exemplu apropiat de seriile reale, cu mai multe componente temporale.</p>
                </div>
            </div>
        </section>
        """,
    )


@synthetic_bp.route("/synthetic/<series_key>")
def synthetic_series_page(series_key: str):
    df_series = generate_synthetic_series(series_key)

    return generic_series_page(
        title=df_series["series_name"].iloc[0],
        df_series=df_series,
        current_path=request.path,
        value_label="Valoare",
        series_kind=f"synthetic_{series_key.replace('-', '_')}",
        theory_key=series_key,
    )


@synthetic_bp.route("/temperature-demo")
def temperature_demo_page():
    df_series = generate_temperature_demo_series()

    return generic_series_page(
        title="Temperatură lunară demonstrativă",
        df_series=df_series,
        current_path=request.path,
        value_label="Temperatură [°C]",
        series_kind="temperature_demo",
        theory_key="temperature_demo",
    )
