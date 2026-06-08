import numpy as np
from statsmodels.tsa.seasonal import STL
from utils.ts_utils import stationarity_metrics_from_series


def detect_trend(series):
    x = np.arange(len(series))
    y = series.values

    slope = np.polyfit(x, y, 1)[0]

    if abs(slope) < 1e-4:
        return "Fără trend semnificativ"
    elif slope > 0:
        return "Trend ascendent"
    else:
        return "Trend descendent"


def detect_seasonality(series, period=12):
    try:
        stl = STL(series, period=period, robust=True).fit()
        seasonal_strength = np.std(stl.seasonal) / np.std(series)

        if seasonal_strength > 0.3:
            return "Sezonalitate puternică"
        elif seasonal_strength > 0.1:
            return "Sezonalitate moderată"
        else:
            return "Fără sezonalitate clară"
    except:
        return "Nu s-a putut determina sezonalitatea"


def detect_variability(series):
    std = series.std()

    if std < 0.1:
        return "Variabilitate scăzută"
    elif std < 0.3:
        return "Variabilitate moderată"
    else:
        return "Variabilitate ridicată"


def generate_insights(series):
    station = stationarity_metrics_from_series(series)

    insights = []

    if station["stationary"]:
        insights.append("Seria este staționară")
    else:
        insights.append("Seria este nestaționară")

    insights.append(detect_trend(series))

    insights.append(detect_seasonality(series))

    insights.append(detect_variability(series))

    return insights