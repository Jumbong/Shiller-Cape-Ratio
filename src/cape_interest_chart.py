"""
cape_interest_chart.py
======================
DRY agent to reproduce the Shiller CAPE P/E10 + Long-Term Interest Rates chart.

Usage
-----
    from cape_interest_chart import plot_cape_interest
    from shiller_loader import load_shiller

    df  = load_shiller("ie_data.xls")
    fig = plot_cape_interest(df)
    fig.savefig("cape_interest.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# Annotation points for CAPE peaks/troughs
# ---------------------------------------------------------------------------
_ANNOTATIONS: list[dict] = [
    {"label": "1901", "date": "1901-06-01", "cape": 25.2, "dx": 0,  "dy":  1.5},
    {"label": "1921", "date": "1920-12-01", "cape":  4.8, "dx": 0,  "dy": -2.5},
    {"label": "1929", "date": "1929-09-01", "cape": 32.6, "dx": 0,  "dy":  1.5},
    {"label": "1966", "date": "1966-01-01", "cape": 24.1, "dx": 0,  "dy":  1.5},
    {"label": "1981", "date": "1980-11-01", "cape":  9.7, "dx": 2,  "dy":  2.5},
    {"label": "2000", "date": "1999-12-01", "cape": 44.2, "dx": 0,  "dy":  1.5},
    {"label": "2022", "date": "2021-11-01", "cape": 38.6, "dx": 1,  "dy":  1.5},
]

_BLUE = "#1f3d7a"
_RED  = "#cc0000"


def plot_cape_interest(
    df: pd.DataFrame,
    start_year: int = 1860,
    end_year:   int = 2040,
    figsize: tuple  = (13, 7),
) -> plt.Figure:
    """
    Reproduce the Shiller CAPE P/E10 + Long-Term Interest Rates dual-axis chart.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_shiller() — must contain 'date', 'cape', 'gs10'.
    start_year : int  Left x-axis limit (default 1860).
    end_year : int    Right x-axis limit (default 2040).
    figsize : tuple   Figure size in inches.

    Returns
    -------
    matplotlib Figure
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    cape_data = df.dropna(subset=["cape"])
    gs10_data = df.dropna(subset=["gs10"])

    latest_cape = cape_data.iloc[-1]["cape"]
    latest_date = cape_data.iloc[-1]["date"]

    # -----------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=figsize)

    # -- 1. CAPE line (left axis — blue solid) ------------------------------
    ax1.plot(cape_data["date"], cape_data["cape"],
             color=_BLUE, linewidth=1.3, zorder=3, label="CAPE")

    # -- 2. Long-Term Interest Rates (right axis — red dashed) --------------
    ax2 = ax1.twinx()
    ax2.plot(gs10_data["date"], gs10_data["gs10"],
             color=_RED, linewidth=1.1, linestyle="--",
             dashes=(4, 2), zorder=2, label="Long-Term Interest Rates")

    # -- 3. Horizontal grid lines (left axis) --------------------------------
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.grid(axis="y", color="lightgrey", linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)

    # -- 4. CAPE annotations (year labels at peaks/troughs) -----------------
    for ann in _ANNOTATIONS:
        ax1.annotate(
            ann["label"],
            xy=(pd.Timestamp(ann["date"]), ann["cape"]),
            xytext=(
                pd.Timestamp(ann["date"]) + pd.DateOffset(years=ann["dx"]),
                ann["cape"] + ann["dy"],
            ),
            fontsize=13, color="black",
            ha="center", va="bottom",
        )

    # -- 5. Latest CAPE value label (bold blue at end of line) --------------
    ax1.annotate(
        f"{latest_cape:.1f}",
        xy=(latest_date, latest_cape),
        xytext=(latest_date + pd.DateOffset(years=1), latest_cape),
        fontsize=13, fontweight="bold", color=_BLUE,
        va="center",
    )

    # -- 6. "CAPE" label on the blue line -----------------------------------
    ax1.annotate(
        "CAPE",
        xy=(pd.Timestamp("1882-01-01"), 13),
        fontsize=12, fontweight="bold", color=_BLUE,
    )

    # -- 7. "Long-Term Interest Rates" label on the red line ---------------
    ax1.annotate(
        "Long-Term\nInterest Rates",
        xy=(pd.Timestamp("1882-01-01"), 4.5),
        fontsize=11, fontweight="bold", color=_RED,
        va="top",
    )

    # -----------------------------------------------------------------------
    # Left axis (CAPE)
    # -----------------------------------------------------------------------
    ax1.set_xlim(pd.Timestamp(f"{start_year}-01-01"),
                 pd.Timestamp(f"{end_year}-01-01"))
    ax1.set_ylim(0, 50)
    ax1.set_ylabel("Price-Earnings Ratio (CAPE, P/E10)",
                   color=_BLUE, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", colors=_BLUE, labelsize=11)
    ax1.tick_params(axis="x", labelsize=11, bottom=False)

    # X axis ticks every 20 years
    ax1.set_xticks([pd.Timestamp(f"{y}-01-01")
                    for y in range(start_year, end_year + 1, 20)])
    ax1.set_xticklabels([str(y)
                         for y in range(start_year, end_year + 1, 20)])

    # -----------------------------------------------------------------------
    # Right axis (GS10 — Long-Term Interest Rates)
    # -----------------------------------------------------------------------
    ax2.set_ylim(0, 18)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax2.set_ylabel("Long-Term Interest Rates",
                   color=_RED, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", colors=_RED, labelsize=11)

    # -----------------------------------------------------------------------
    # Remove spines for cleaner look
    # -----------------------------------------------------------------------
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    ax1.set_title("CAPE Price E10 Ratio", fontsize=13, color="black", pad=12)

    plt.tight_layout()
    return fig


