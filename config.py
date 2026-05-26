from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "indices_timeseries.csv"

ROI_INFO = {

    "roi1": {
        "label": "ROI 1",
        "category": "Regiune agricolă / vegetativă",
        "description": (
            "Regiune extrasă din cube-ul satelitar Sentinel-2 "
            "utilizată pentru analiza spațio-temporală "
            "a indicilor spectrali."
        ),
        "coords": "Date Sentinel-2",
        "expected_ndvi": (
            "Variabilitate sezonieră ridicată"
        ),
        "route": "/spectral-indices?roi=roi1",
    },

    "roi2": {
        "label": "ROI 2",
        "category": "Regiune mixtă",
        "description": (
            "Regiune utilizată pentru compararea "
            "semnăturilor spectrale și detectarea "
            "anomaliilor vegetative."
        ),
        "coords": "Date Sentinel-2",
        "expected_ndvi": (
            "Variabilitate spectrală moderată"
        ),
        "route": "/spectral-indices?roi=roi2",
    },

}

SITE_LABELS = {
    "roi1": "ROI 1",
    "roi2": "ROI 2",
}

NAV_ITEMS = [

    {
        "label": "Acasă",
        "href": "/",
    },

    {
        "label": "Catalog serii",
        "href": "/series-catalog",
    },

    {
        "label": "Date satelitare",
        "children": [

            {
                "label": "ROI-uri",
                "href": "/roi",
            },

            {
                "label": "Indici spectrali",
                "href": "/spectral-indices",
            },

            {
                "label": "Cross-Index",
                "href": "/cross-index-analysis",
            },

        ]
    },

    {
        "label": "Analiză temporală",
        "children": [

            {
                "label": "Staționaritate",
                "href": "/stationarity",
            },

            {
                "label": "STL Decompose",
                "href": "/decompose",
            },

            {
                "label": "Trend",
                "href": "/trend",
            },

            {
                "label": "Sezonalitate",
                "href": "/seasonality",
            },

            {
                "label": "Anomalii",
                "href": "/anomalies",
            },

        ]
    },

    {
        "label": "ML & Clustering",
        "children": [

            {
                "label": "ML Features",
                "href": "/ml-features",
            },

        ]
    },

    {
        "label": "Forecast",
        "children": [

            {
                "label": "Forecast ARIMA",
                "href": "/forecast-arima",
            },

            {
                "label": "Forecast LSTM",
                "href": "/forecast-lstm",
            },

        ]
    },

    {
        "label": "Validare",
        "children": [

            {
                "label": "White Noise",
                "href": "/synthetic/white-noise",
            },

            {
                "label": "Random Walk",
                "href": "/synthetic/random-walk",
            },

            {
                "label": "Temperatură",
                "href": "/temperature-demo",
            },

        ]
    },

    {
        "label": "Metodologie",
        "href": "/methodology",
    },

]

HOME_SECTIONS = [

    {
        "title": "Date Sentinel-2",
        "description": (
            "Analiză spațio-temporală pe ROI-uri "
            "și indici spectrali Sentinel-2."
        ),
        "links": [
            {
                "label": "ROI-uri",
                "href": "/roi",
            },
            {
                "label": "Indici spectrali",
                "href": "/spectral-indices",
            },
            {
                "label": "Cross-Index",
                "href": "/cross-index-analysis",
            },
        ],
    },

    {
        "title": "Analiză temporală",
        "description": (
            "Trend, sezonalitate, STL decomposition "
            "și detectarea anomaliilor."
        ),
        "links": [
            {
                "label": "Trend",
                "href": "/trend",
            },
            {
                "label": "STL",
                "href": "/decompose",
            },
            {
                "label": "ADF",
                "href": "/stationarity",
            },
        ],
    },

    {
        "title": "ML & Clustering",
        "description": (
            "Reducere dimensională, clustering, "
            "Isolation Forest și DTW."
        ),
        "links": [
            {
                "label": "ML Features",
                "href": "/ml-features",
            },
        ],
    },

    {
        "title": "Forecast",
        "description": (
            "Predicția seriilor temporale "
            "folosind ARIMA și LSTM."
        ),
        "links": [
            {
                "label": "Forecast ARIMA",
                "href": "/forecast-arima",
            },
            {
                "label": "Forecast LSTM",
                "href": "/forecast-lstm",
            },
        ],
    },

]

SERIES_CATALOG = [

    {
        "group": "Serii sintetice",
        "description": (
            "Serii utilizate pentru validarea "
            "pipeline-ului de analiză temporală."
        ),
        "items": [
            {
                "label": "White Noise",
                "href": "/synthetic/white-noise",
                "tag": "Staționară"
            },

            {
                "label": "Random Walk",
                "href": "/synthetic/random-walk",
                "tag": "Nestaționară"
            },

            {
                "label": "Trend liniar + zgomot",
                "href": "/synthetic/linear-trend",
                "tag": "Trend"
            },

            {
                "label": "Sinusoidală + zgomot",
                "href": "/synthetic/seasonal-noise",
                "tag": "Sezonieră"
            },

            {
                "label": "Trend + sezonalitate",
                "href": "/synthetic/trend-seasonal",
                "tag": "Completă"
            },

        ],
    },

    {
        "group": "Serii climatice",
        "description": (
            "Date demonstrative pentru validarea "
            "generalizării metodologiei."
        ),
        "items": [

            {
                "label": "Temperatură lunară",
                "href": "/temperature-demo",
                "tag": "Climatică"
            },

        ],
    },

    {
        "group": "Serii Sentinel-2",
        "description": (
            "Date reale Sentinel-2 utilizate "
            "pentru analiza spațio-temporală."
        ),
        "items": [

            {
                "label": "ROI-uri",
                "href": "/roi",
                "tag": "Sentinel-2"
            },

            {
                "label": "Indici spectrali",
                "href": "/spectral-indices",
                "tag": "Indices"
            },

            {
                "label": "ML Features",
                "href": "/ml-features",
                "tag": "ML"
            },

        ],
    },

]