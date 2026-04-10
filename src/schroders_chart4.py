"""
schroders_chart4.py
===================
DRY agent to reproduce Schroders Chart 4:
"Lows in the 12-month trailing S&P P/E ratio during recessions"

Source: Refinitiv, Robert Shiller, Schroders Economics Group, 5 December 2022.

Usage
-----
    from schroders_chart4 import plot_chart4
    from shiller_loader import load_shiller

    df = load_shiller("ie_data.xls")
    fig = plot_chart4(df)
    fig.show()
    fig.write_html("chart4.html")
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# NBER recession dates — exactly matching the Schroders chart x-axis labels
# ---------------------------------------------------------------------------
NBER_RECESSIONS: list[tuple[str, str, str]] = [
    # (start,        end,          x-axis label)
    ("1929-08-01", "1933-03-01", "Aug 1929-Mar 1933"),
    ("1937-05-01", "1938-06-01", "May 1937-Jun 1938"),
    ("1945-02-01", "1945-10-01", "Feb-Oct 1945"),
    ("1948-11-01", "1949-10-01", "Nov 1948-Oct 1949"),
    ("1953-07-01", "1954-05-01", "Jul 1953-May 1957"),
    ("1957-08-01", "1958-04-01", "Aug 1957-Apr 1958"),
    ("1960-04-01", "1961-02-01", "Apr 1960-Feb 1961"),
    ("1969-12-01", "1970-11-01", "Dec 1969-Nov 1970"),
    ("1973-11-01", "1975-03-01", "Nov 1973-Mar 1975"),
    ("1980-01-01", "1980-07-01", "Jan-Jul 1980"),
    ("1981-07-01", "1982-11-01", "Jul 1981-Nove 1982"),
    ("1990-07-01", "1991-03-01", "Jul 1990-Mar 1991"),
    ("2001-03-01", "2001-11-01", "Mar-Nov 2001"),
    ("2007-12-01", "2009-06-01", "Dec 2007-Jun 2009"),
    ("2020-02-01", "2020-04-01", "Feb-Apr 2020"),
]

# Schroders chart reference date (Dec 2022)
_LATEST_DATE = "2022-12-01"

# Colours matching Schroders style
_BAR_COLOUR    = "#1a2f5e"   # dark navy blue
_AVG_COLOUR    = "#4caf50"   # green
_LATEST_COLOUR = "#00bcd4"   # light blue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_recession_pe_lows(
    df: pd.DataFrame,
    recessions: list[tuple[str, str, str]] = NBER_RECESSIONS,
    latest_date: str = _LATEST_DATE,
) -> pd.DataFrame:
    """
    Compute the minimum trailing 12-month P/E during each recession.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_shiller() — must contain 'date', 'real_price', 'real_earnings'.
    recessions : list of (start, end, label)
        NBER recession periods.
    latest_date : str
        Reference date for the "Latest" horizontal line.

    Returns
    -------
    pd.DataFrame with columns: label, pe_min, start, end
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Trailing 12-month P/E: price divided by 12-month rolling average of earnings
    # More accurate than simple price/earnings since real_earnings is already annualized
    # by Shiller via linear interpolation of quarterly data
    df["earnings_12m_avg"] = df["real_earnings"].rolling(window=12).mean()
    df["pe"]               = df["real_price"] / df["earnings_12m_avg"]

    rows = []
    for start, end, label in recessions:
        mask   = (df["date"] >= start) & (df["date"] <= end)
        pe_min = df.loc[mask, "pe"].min()
        if not np.isnan(pe_min):
            rows.append({"label": label, "pe_min": pe_min,
                         "start": start, "end": end})

    result = pd.DataFrame(rows)

    # Latest PE value
    latest_mask = df["date"] == pd.Timestamp(latest_date)
    result.attrs["pe_latest"] = float(df.loc[latest_mask, "pe"].values[0]) \
        if latest_mask.any() else df["pe"].dropna().iloc[-1]
    result.attrs["pe_avg"]    = result["pe_min"].mean()

    return result


def plot_chart4(
    df: pd.DataFrame,
    recessions: list[tuple[str, str, str]] = NBER_RECESSIONS,
    latest_date: str = _LATEST_DATE,
    title: str = "Chart 4: Lows in the 12-month trailing S&P P/E<br>ratio during recessions",
) -> go.Figure:
    """
    Reproduce Schroders Chart 4 as an interactive Plotly figure.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_shiller().
    recessions : list of (start, end, label)
        NBER recession periods. Defaults to NBER_RECESSIONS.
    latest_date : str
        Reference date for the "Latest" line (default Dec 2022).
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    data       = compute_recession_pe_lows(df, recessions, latest_date)
    pe_avg     = data.attrs["pe_avg"]
    pe_latest  = data.attrs["pe_latest"]

    fig = go.Figure()

    # -- 1. Bars -----------------------------------------------------------
    fig.add_trace(go.Bar(
        x         = data["label"],
        y         = data["pe_min"],
        name      = "Low in the S&P 12-month trailing P/E ratios",
        marker_color = _BAR_COLOUR,
        hovertemplate = "<b>%{x}</b><br>P/E low: %{y:.1f}<extra></extra>",
    ))

    # -- 2. Average horizontal line ----------------------------------------
    fig.add_hline(
        y          = pe_avg,
        line_color = _AVG_COLOUR,
        line_width = 2,
        annotation_text     = f"Average ({pe_avg:.1f})",
        annotation_position = "top right",
        annotation_font     = dict(color=_AVG_COLOUR, size=10),
    )

    # -- 3. Latest horizontal line -----------------------------------------
    fig.add_hline(
        y          = pe_latest,
        line_color = _LATEST_COLOUR,
        line_width = 2,
        annotation_text     = f"Latest ({pe_latest:.1f})",
        annotation_position = "top left",
        annotation_font     = dict(color=_LATEST_COLOUR, size=10),
    )

    # -- 4. Layout ---------------------------------------------------------
    fig.update_layout(
        title      = dict(text=title, font=dict(size=13, color="#1a2f5e"), x=0),
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        yaxis = dict(
            range     = [0, 25],
            tickvals  = [0, 5, 10, 15, 20, 25],
            showgrid  = True,
            gridcolor = "rgba(200,200,200,0.3)",
            zeroline  = False,
        ),
        xaxis = dict(
            tickangle = -45,
            showgrid  = False,
        ),
        legend = dict(
            orientation = "h",
            y           = -0.35,
            x           = 0,
            font        = dict(size=10),
        ),
        margin = dict(l=50, r=30, t=70, b=150),
        width  = 600,
        height = 520,
        showlegend = True,
    )

    return fig


