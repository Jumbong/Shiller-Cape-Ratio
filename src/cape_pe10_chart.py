"""
cape_pe10_chart.py
==================
DRY agent to reproduce the Shiller CAPE P/E10 Ratio chart.

Usage
-----
    from cape_pe10_chart import plot_cape_pe10
    from shiller_loader import load_shiller

    df = load_shiller("ie_data.xls")
    fig = plot_cape_pe10(df)
    fig.savefig("cape_pe10.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Annotation points — (year_label, peak/trough date, x_offset, y_offset)
# ---------------------------------------------------------------------------
_ANNOTATIONS: list[dict] = [
    {"label": "1901", "date": "1901-06-01", "cape": 25.2, "dx": 0,   "dy":  1.5},
    {"label": "1921", "date": "1920-12-01", "cape":  4.8, "dx": 0,   "dy": -2.5},
    {"label": "1929", "date": "1929-09-01", "cape": 32.6, "dx": 0,   "dy":  1.5},
    {"label": "1966", "date": "1966-01-01", "cape": 24.1, "dx": 0,   "dy":  1.5},
    {"label": "1981", "date": "1980-11-01", "cape":  9.7, "dx": 2,   "dy":  2.5},
    {"label": "2000", "date": "1999-12-01", "cape": 44.2, "dx": 0,   "dy":  1.5},
    {"label": "2022", "date": "2021-11-01", "cape": 38.6, "dx": 1,   "dy":  1.5},
]

_BLUE = "#1f3d7a"


def plot_cape_pe10(
    df: pd.DataFrame,
    start_year: int = 1860,
    end_year:   int = 2040,
    figsize: tuple = (13, 7),
) -> plt.Figure:
    """
    Reproduce the Shiller CAPE P/E10 Ratio chart.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_shiller() — must contain 'date' and 'cape'.
    start_year : int
        Left x-axis limit (default 1860).
    end_year : int
        Right x-axis limit (default 2040).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib Figure
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["cape"]).sort_values("date").reset_index(drop=True)

    # Latest value for the end-of-line label
    latest      = df.iloc[-1]
    latest_date = latest["date"]
    latest_cape = latest["cape"]

    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # -- 1. CAPE line -------------------------------------------------------
    ax.plot(df["date"], df["cape"],
            color=_BLUE, linewidth=1.2, zorder=3)

    # -- 2. Horizontal grid lines -------------------------------------------
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(axis="y", color="lightgrey", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # -- 3. Annotations (year labels at peaks/troughs) ----------------------
    for ann in _ANNOTATIONS:
        ax.annotate(
            ann["label"],
            xy=(pd.Timestamp(ann["date"]), ann["cape"]),
            xytext=(pd.Timestamp(ann["date"]) + pd.DateOffset(years=ann["dx"]),
                    ann["cape"] + ann["dy"]),
            fontsize=13, fontweight="normal",
            color="black",
            ha="center", va="bottom",
        )

    # -- 4. Latest value label (bold blue at end of line) -------------------
    ax.annotate(
        f"{latest_cape:.1f}",
        xy=(latest_date, latest_cape),
        xytext=(latest_date + pd.DateOffset(years=1), latest_cape),
        fontsize=13, fontweight="bold", color=_BLUE,
        va="center",
    )

    # -- 5. "CAPE" label on the line (left side) ----------------------------
    ax.annotate(
        "CAPE",
        xy=(pd.Timestamp("1882-01-01"), 13),
        fontsize=13, fontweight="bold", color=_BLUE,
    )

    # -----------------------------------------------------------------------
    # Axes formatting
    # -----------------------------------------------------------------------
    ax.set_xlim(pd.Timestamp(f"{start_year}-01-01"),
                pd.Timestamp(f"{end_year}-01-01"))
    ax.set_ylim(0, 50)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20 * 365.25))
    ax.xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%Y")
    )
    ax.set_xticks([
        pd.Timestamp(f"{y}-01-01")
        for y in range(start_year, end_year + 1, 20)
    ])
    ax.set_xticklabels(
        [str(y) for y in range(start_year, end_year + 1, 20)],
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=11)

    ax.set_ylabel("Price-Earnings Ratio (CAPE, P/E10)",
                  color=_BLUE, fontsize=12, fontweight="bold")
    ax.yaxis.label.set_color(_BLUE)
    ax.tick_params(axis="y", colors=_BLUE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", bottom=False)

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    ax.set_title("CAPE Price E10 Ratio", fontsize=13, color="black", pad=12)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
