"""
cape_crisis_chart.py
====================
DRY agent to reproduce the CAPE ratio chart highlighting peaks before major crises.

Usage
-----
    from cape_crisis_chart import plot_cape_crises
    from shiller_loader import load_shiller

    df = load_shiller("ie_data.xls")
    fig = plot_cape_crises(df)
    fig.show()
    fig.write_html("cape_crises.html")
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Crisis peaks — (date, label, annotation position)
# ---------------------------------------------------------------------------
CRISIS_PEAKS: list[dict] = [
    {
        "date":        "1929-09-01",
        "cape":        32.6,
        "label":       "1929<br>Great Depression",
        "ax": -40, "ay": -50,
    },
    {
        "date":        "1999-12-01",
        "cape":        44.2,
        "label":       "2000<br>Dot-com Bubble",
        "ax": 40, "ay": -50,
    },
    {
        "date":        "2007-05-01",
        "cape":        27.5,
        "label":       "2007<br>Financial Crisis",
        "ax": 50, "ay": -45,
    },
    {
        "date":        "2026-01-01",
        "cape":        39.6,
        "label":       "Today<br>(Apr 2026)",
        "ax": -80, "ay": -50,
    },
]

_BLUE      = "#1a4f8a"
_RED       = "#d32f2f"
_LIGHT_GREY = "rgba(220,220,220,0.4)"


def plot_cape_crises(
    df: pd.DataFrame,
    start: str = "1881-01-01",
    end:   str = "2026-04-01",
    title: str = "",
) -> go.Figure:
    """
    Plot the Shiller CAPE ratio from 1881 to today,
    highlighting peaks before major crises with red circles and annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_shiller().
    start : str
        Start date (default 1881-01-01).
    end : str
        End date (default 2026-04-01).
    title : str
        Optional figure title.

    Returns
    -------
    go.Figure
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["cape"])
    mask = (df["date"] >= start) & (df["date"] <= end)
    d = df[mask].reset_index(drop=True)

    fig = go.Figure()

    # -- 1. CAPE line -------------------------------------------------------
    fig.add_trace(go.Scatter(
        x    = d["date"],
        y    = d["cape"],
        mode = "lines",
        name = "CAPE Ratio",
        line = dict(color=_BLUE, width=1.5),
        hovertemplate = "%{x|%Y-%m}<br>CAPE: %{y:.1f}<extra></extra>",
    ))

    # -- 2. Historical mean line --------------------------------------------
    cape_mean = d["cape"].mean()
    fig.add_hline(
        y          = cape_mean,
        line_color = "rgba(100,100,100,0.5)",
        line_dash  = "dash",
        line_width = 1,
        annotation_text     = f"Historical mean: {cape_mean:.0f}",
        annotation_position = "bottom right",
        annotation_font     = dict(color="grey", size=10),
    )

    # -- 3. Red circles + annotations at crisis peaks ----------------------
    for crisis in CRISIS_PEAKS:
        date = pd.Timestamp(crisis["date"])
        cape = crisis["cape"]
        label = crisis["label"]

        # Red circle marker
        fig.add_trace(go.Scatter(
            x    = [date],
            y    = [cape],
            mode = "markers",
            name = label.replace("<br>", " — "),
            marker = dict(
                color  = _RED,
                size   = 14,
                line   = dict(color="white", width=2),
                symbol = "circle",
            ),
            hovertemplate = f"<b>{label.replace('<br>',' ')}</b><br>"
                            f"CAPE: {cape:.1f}<extra></extra>",
            showlegend = True,
        ))

        # Annotation arrow + label
        fig.add_annotation(
            x          = date,
            y          = cape,
            text       = f"<b>{label}</b><br>CAPE: {cape:.0f}",
            showarrow  = True,
            arrowhead  = 2,
            arrowcolor = _RED,
            arrowwidth = 1.5,
            ax         = crisis["ax"],
            ay         = crisis["ay"],
            font       = dict(color=_RED, size=10),
            bgcolor    = "rgba(255,255,255,0.85)",
            bordercolor = _RED,
            borderwidth = 1,
            borderpad  = 4,
        )

    # -- 4. Shade "extreme" zone (above 30) --------------------------------
    fig.add_hrect(
        y0        = 30, y1 = 50,
        fillcolor = "rgba(211,47,47,0.07)",
        line_width = 0,
        annotation_text     = "Extreme zone (CAPE > 30)",
        annotation_position = "top left",
        annotation_font     = dict(color=_RED, size=9),
    )

    # -- 5. Layout ---------------------------------------------------------
    fig.update_layout(
        title = dict(
            text = title,
            font = dict(size=13, color="#222"),
            x    = 0,
        ),
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        hovermode     = "x unified",
        xaxis = dict(
            showgrid   = False,
            tickformat = "%Y",
            dtick      = "M240",
            tickangle  = 0,
            range      = [pd.Timestamp(start), pd.Timestamp('2027-01-01')],
        ),
        yaxis = dict(
            range     = [0, 50],
            tickvals  = list(range(0, 51, 10)),
            showgrid  = True,
            gridcolor = "rgba(200,200,200,0.3)",
            zeroline  = False,
            title     = "CAPE Ratio",
        ),
        legend = dict(
            orientation = "h",
            y           = -0.15,
            x           = 0,
            font        = dict(size=10),
        ),
        margin = dict(l=60, r=120, t=60, b=80),
        width  = 900,
        height = 500,
    )

    return fig
