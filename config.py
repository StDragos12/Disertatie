from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ndvi_timeseries_csv_multi.csv"

ROI_INFO = {
    "ParcBucuresti": {
        "label": "Parc București",
        "category": "Spațiu verde urban / parc",
        "description": (
            "ROI reprezentativ pentru vegetație urbană densă. "
            "Este utilizat pentru evidențierea unui comportament NDVI mai ridicat, "
            "cu sezonalitate moderată."
        ),
        "coords": "Poligon definit în setul de date sursă",
        "expected_ndvi": "Ridicat spre mediu-ridicat, cu vârfuri în sezonul cald",
        "route": "/padure",
    },
    "AgricolIlfov": {
        "label": "Agricol Ilfov",
        "category": "Teren agricol",
        "description": (
            "ROI selectat pentru a surprinde dinamica vegetației agricole. "
            "Prezintă amplitudini sezoniere mai mari și variații corelate cu ciclurile de vegetație."
        ),
        "coords": "Poligon definit în setul de date sursă",
        "expected_ndvi": "Variabil, cu amplitudine sezonieră mare",
        "route": "/agricol",
    },
    "UrbanCentral": {
        "label": "Urban Central",
        "category": "Țesut urban dens",
        "description": (
            "ROI reprezentativ pentru o zonă urbană cu vegetație redusă. "
            "Este util pentru comparația cu zonele verzi și cele agricole."
        ),
        "coords": "Poligon definit în setul de date sursă",
        "expected_ndvi": "Scăzut, cu variații mai reduse",
        "route": "/urban",
    },
}

SITE_LABELS = {
    "ParcBucuresti": "Parc București",
    "UrbanCentral": "Urban Central",
    "AgricolIlfov": "Agricol Ilfov",
}

NAV_ITEMS = [
    {"label": "Acasă", "href": "/"},
    {"label": "Catalog serii", "href": "/series-catalog"},
    {"label": "ROI", "href": "/roi"},
    {
        "label": "Analiză",
        "children": [
            {"label": "Statistici NDVI", "href": "/stats"},
            {"label": "Compare Metrics", "href": "/compare-metrics"},
            {"label": "Staționaritate", "href": "/stationarity"},
            {"label": "Decompose", "href": "/decompose"},
            {"label": "Trend", "href": "/trend"},
            {"label": "Sezonalitate", "href": "/seasonality"},
            {"label": "Anomalii", "href": "/anomalies"},
        ],
    },
    {
        "label": "Forecast",
        "children": [
            {"label": "Forecast ARIMA", "href": "/forecast-arima"},
            {"label": "Forecast LSTM", "href": "/forecast-lstm"},
        ],
    },
    {
        "label": "Seturi de date",
        "children": [
            {"label": "NDVI – toate siturile", "href": "/toate"},
            {"label": "Parc București", "href": "/padure"},
            {"label": "Agricol Ilfov", "href": "/agricol"},
            {"label": "Urban Central", "href": "/urban"},
            {"label": "White Noise", "href": "/synthetic/white-noise"},
            {"label": "Random Walk", "href": "/synthetic/random-walk"},
            {"label": "Trend liniar", "href": "/synthetic/linear-trend"},
            {"label": "Sinusoidală", "href": "/synthetic/seasonal-noise"},
            {"label": "Trend + sezonalitate", "href": "/synthetic/trend-seasonal"},
            {"label": "Temperatură demonstrativă", "href": "/temperature-demo"},
        ],
    },
    {"label": "Metodologie", "href": "/methodology"},
]

HOME_SECTIONS = [
    {
        "title": "Serii sintetice",
        "description": (
            "Seturi de date controlate pentru demonstrarea conceptelor fundamentale din analiza seriilor temporale."
        ),
        "links": [
            {"label": "White Noise", "href": "/synthetic/white-noise"},
            {"label": "Random Walk", "href": "/synthetic/random-walk"},
            {"label": "Trend + sezonalitate", "href": "/synthetic/trend-seasonal"},
        ],
    },
    {
        "title": "Serii climatice / de mediu",
        "description": (
            "Exemple non-satelitare pentru a arăta generalizarea metodologiei."
        ),
        "links": [
            {"label": "Temperatură lunară demonstrativă", "href": "/temperature-demo"},
            {"label": "Catalog serii", "href": "/series-catalog"},
        ],
    },
    {
        "title": "Serii NDVI Sentinel-2",
        "description": (
            "Studiul principal de caz al aplicației, bazat pe trei ROI-uri cu acoperire diferită."
        ),
        "links": [
            {"label": "Compară toate", "href": "/toate"},
            {"label": "ROI", "href": "/roi"},
            {"label": "Forecast ARIMA", "href": "/forecast-arima"},
        ],
    },
    {
        "title": "Instrumente analitice",
        "description": (
            "Statistici descriptive, ADF, STL, anomalii și forecasting."
        ),
        "links": [
            {"label": "Statistici", "href": "/stats"},
            {"label": "Staționaritate", "href": "/stationarity"},
            {"label": "Decompose", "href": "/decompose"},
            {"label": "Anomalii", "href": "/anomalies"},
        ],
    },
]

SERIES_CATALOG = [
    {
        "group": "Serii sintetice",
        "description": "Exemple controlate pentru ilustrarea conceptelor de staționaritate, trend și sezonalitate.",
        "items": [
            {"label": "White Noise", "href": "/synthetic/white-noise", "tag": "Staționară"},
            {"label": "Random Walk", "href": "/synthetic/random-walk", "tag": "Nestaționară"},
            {"label": "Trend liniar + zgomot", "href": "/synthetic/linear-trend", "tag": "Trend"},
            {"label": "Sinusoidală + zgomot", "href": "/synthetic/seasonal-noise", "tag": "Sezonieră"},
            {"label": "Trend + sezonalitate", "href": "/synthetic/trend-seasonal", "tag": "Completă"},
        ],
    },
    {
        "group": "Serii climatice demonstrative",
        "description": "Serii non-satelitare pentru a arăta că metodologia poate fi aplicată și altor domenii.",
        "items": [
            {"label": "Temperatură lunară demonstrativă", "href": "/temperature-demo", "tag": "Climatică"},
        ],
    },
    {
        "group": "Serii NDVI Sentinel-2",
        "description": "Studiul principal de caz al aplicației, bazat pe trei ROI-uri distincte.",
        "items": [
            {"label": "Parc București", "href": "/padure", "tag": "NDVI"},
            {"label": "Agricol Ilfov", "href": "/agricol", "tag": "NDVI"},
            {"label": "Urban Central", "href": "/urban", "tag": "NDVI"},
            {"label": "Compară toate", "href": "/toate", "tag": "Dashboard"},
        ],
    },
]