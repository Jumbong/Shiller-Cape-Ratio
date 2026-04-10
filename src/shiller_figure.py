"""
shiller_figure1.py
==================
DRY agent to reproduce Shiller Figure 1 with train/test split:
  - Train (1881–2004): solid lines
  - Test  (2005+)    : dotted lines + grey shaded area
  - Actual Returns + Forecast Returns (fitted/predicted) + CAPE Ratio

Usage
-----
    from shiller_figure1 import plot_shiller_figure1
    from shiller_loader import load_shiller
    from ols_lm import lm
    import numpy as np, pandas as pd

    df = load_shiller("ie_data.xls")
    df["date"] = pd.to_datetime(df["date"])

    d_train = df[(df["date"] >= "1881-01-01") & (df["date"] <= "2004-12-01")].dropna(subset=["cape","real_10y_stock_return"])
    d_test  = df[df["date"] >= "2005-01-01"].dropna(subset=["cape"])

    result = lm(
        y          = d_train["real_10y_stock_return"] * 100,
        X          = np.log(d_train["cape"]),
        y_name     = "RET_t",
        x_names    = ["log(CAPE_t)"],
        X_forecast = np.log(d_test["cape"]),
    )

    fig = plot_shiller_figure1(
        dates_train   = d_train["date"],
        actual_train  = d_train["real_10y_stock_return"] * 100,
        fitted_train  = result["fitted"],
        dates_test    = d_test["date"],
        actual_test   = d_test["real_10y_stock_return"] * 100,
        forecast_test = result["forecast"],
        cape          = pd.concat([d_train["cape"], d_test["cape"]]),
        dates_cape    = pd.concat([d_train["date"], d_test["date"]]),
    )
    fig.show()
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_DARK_BLUE  = "#1a4f8a"   # train actual returns
_ORANGE     = "#e65100"   # test actual returns
_GREEN      = "#2e7d32"   # fitted / forecast returns
_LIGHT_BLUE = "#7ba7d4"   # CAPE ratio
_GREY_LINE  = "rgba(180,180,180,0.6)"
_GREY_SHADE = "rgba(200,200,200,0.18)"


def plot_shiller_figure1(
    dates_train:   pd.Series,
    actual_train:  pd.Series,
    fitted_train:  pd.Series,
    dates_test:    pd.Series,
    actual_test:   pd.Series,
    forecast_test: pd.Series,
    cape:          pd.Series,
    dates_cape:    pd.Series,
    split_date:    str = "2005-01-01",
    title: str = (
        "Shiller CAPE Ratio — Actual vs Forecast 10-Year Real Stock Returns<br>"
        "<sup>Train: 1881–2004 (solid) | Test: 2005+ (dotted) | Grey zone = out-of-sample</sup>"
    ),
    left_range:  list[float] = [-10, 25],
    right_range: list[float] = [-10, 60],
) -> go.Figure:
    """
    Plot Actual and Forecast Returns with train/test visual split.

    Parameters
    ----------
    dates_train   : Dates for the training period.
    actual_train  : Actual 10Y returns in % — train period.
    fitted_train  : Fitted (in-sample) forecast — train period.
    dates_test    : Dates for the test period.
    actual_test   : Actual 10Y returns in % — test (NaN for most recent obs).
    forecast_test : Out-of-sample forecast from lm() — test period.
    cape          : Full CAPE ratio series (train + test).
    dates_cape    : Dates matching cape series.
    split_date    : Date of the train/test cut (default '2005-01-01').
    title         : Figure title.
    left_range    : Y range for returns axis.
    right_range   : Y range for CAPE axis.

    Returns
    -------
    go.Figure
    """
    # Reset all indexes for consistent alignment
    dates_train   = pd.to_datetime(dates_train).reset_index(drop=True)
    dates_test    = pd.to_datetime(dates_test).reset_index(drop=True)
    dates_cape    = pd.to_datetime(dates_cape).reset_index(drop=True)
    actual_train  = pd.Series(actual_train).reset_index(drop=True)
    fitted_train  = pd.Series(fitted_train).reset_index(drop=True)
    actual_test   = pd.Series(actual_test).reset_index(drop=True)
    forecast_test = pd.Series(forecast_test).reset_index(drop=True)
    cape          = pd.Series(cape).reset_index(drop=True)
    cape_mean     = cape.dropna().mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # -----------------------------------------------------------------------
    # 1. Grey shaded zone for test / out-of-sample period
    # -----------------------------------------------------------------------
    fig.add_vrect(
        x0         = pd.Timestamp(split_date),
        x1         = dates_test.iloc[-1] + pd.DateOffset(months=6),
        fillcolor  = _GREY_SHADE,
        line_width = 0,
        annotation_text     = "Out-of-sample (Test)",
        annotation_position = "top left",
        annotation_font     = dict(color="grey", size=10),
    )

    # -----------------------------------------------------------------------
    # 2. Actual Returns — TRAIN (solid dark blue)
    # -----------------------------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = dates_train,
        y    = actual_train,
        name = "Actual Returns — Train",
        mode = "lines",
        line = dict(color=_DARK_BLUE, width=1.4),
        hovertemplate = "<b>Actual (Train)</b><br>%{x|%Y-%m}<br>%{y:.2f}%<extra></extra>",
    ), secondary_y=False)

    # -----------------------------------------------------------------------
    # 3. Actual Returns — TEST (dotted orange)
    # -----------------------------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = dates_test,
        y    = actual_test,
        name = "Actual Returns — Test",
        mode = "lines",
        line = dict(color=_ORANGE, width=1.4, dash="dot"),
        hovertemplate = "<b>Actual (Test)</b><br>%{x|%Y-%m}<br>%{y:.2f}%<extra></extra>",
    ), secondary_y=False)

    # -----------------------------------------------------------------------
    # 4. Forecast Returns — TRAIN fitted (solid green)
    # -----------------------------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = dates_train,
        y    = fitted_train,
        name = "Forecast Returns — Train (fitted)",
        mode = "lines",
        line = dict(color=_GREEN, width=1.2),
        hovertemplate = "<b>Forecast (Train)</b><br>%{x|%Y-%m}<br>%{y:.2f}%<extra></extra>",
    ), secondary_y=False)

    # -----------------------------------------------------------------------
    # 5. Forecast Returns — TEST out-of-sample (dotted green)
    # -----------------------------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = dates_test,
        y    = forecast_test,
        name = "Forecast Returns — Test (OOS)",
        mode = "lines",
        line = dict(color=_GREEN, width=1.2, dash="dot"),
        hovertemplate = "<b>Forecast (Test)</b><br>%{x|%Y-%m}<br>%{y:.2f}%<extra></extra>",
    ), secondary_y=False)

    # -----------------------------------------------------------------------
    # 6. CAPE Ratio — right axis (light blue)
    # -----------------------------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = dates_cape,
        y    = cape,
        name = "CAPE Ratio",
        mode = "lines",
        line = dict(color=_LIGHT_BLUE, width=1.0),
        hovertemplate = "<b>CAPE</b><br>%{x|%Y-%m}<br>%{y:.1f}<extra></extra>",
    ), secondary_y=True)

    # CAPE historical mean
    fig.add_trace(go.Scatter(
        x    = [dates_cape.iloc[0], dates_cape.iloc[-1]],
        y    = [cape_mean, cape_mean],
        name = f"CAPE Mean ({cape_mean:.1f})",
        mode = "lines",
        line = dict(color=_DARK_BLUE, width=1.0, dash="dash"),
        hovertemplate = f"<b>CAPE Mean</b>: {cape_mean:.1f}<extra></extra>",
    ), secondary_y=True)

    # Zero line on left axis
    fig.add_hline(y=0, line=dict(color=_GREY_LINE, width=0.8), secondary_y=False)

    # -----------------------------------------------------------------------
    # 7. Annotations
    # -----------------------------------------------------------------------
    annotations = [
        dict(x=pd.Timestamp("1940-01-01"), y=19,
             text="<b>Actual Returns</b>",
             showarrow=False, xref="x", yref="y",
             font=dict(color=_DARK_BLUE, size=11)),
        dict(x=pd.Timestamp("1940-01-01"), y=13,
             text="<b>Forecast Returns</b>",
             showarrow=False, xref="x", yref="y",
             font=dict(color=_GREEN, size=11)),
        dict(x=pd.Timestamp("1925-01-01"), y=38,
             text="<b>CAPE Ratio</b>",
             showarrow=False, xref="x", yref="y2",
             font=dict(color=_LIGHT_BLUE, size=11)),
        dict(x=pd.Timestamp("1985-01-01"), y=cape_mean + 2,
             text="Mean",
             showarrow=False, xref="x", yref="y2",
             font=dict(color=_DARK_BLUE, size=10)),
    ]

    # -----------------------------------------------------------------------
    # 8. Layout
    # -----------------------------------------------------------------------
    fig.update_layout(
        title         = dict(text=title, font=dict(size=12, color="#222"), x=0),
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        hovermode     = "x unified",
        annotations   = annotations,
        legend        = dict(orientation="h", y=-0.20, x=0, font=dict(size=10)),
        margin        = dict(l=60, r=80, t=90, b=110),
        width=950, height=560,
    )

    fig.update_yaxes(
        title_text = "Real Stock Returns (%)",
        range      = left_range,
        tickvals   = list(range(-10, 26, 5)),
        showgrid   = True, gridcolor="rgba(200,200,200,0.3)",
        zeroline   = False, secondary_y=False,
    )
    fig.update_yaxes(
        title_text = "CAPE Ratio",
        range      = right_range,
        tickvals   = [10, 30, 50],
        showgrid   = False, secondary_y=True,
    )
    fig.update_xaxes(
        showgrid    = False,
        tickformat  = "%Y",
        dtick       = "M240",
        tickangle   = 0,
    )

    return fig


