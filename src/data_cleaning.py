"""
shiller_loader.py
=================
DRY agent to load and format Shiller's ie_data.xls into a clean pandas DataFrame.

Usage
-----
    from shiller_loader import load_shiller

    df = load_shiller("ie_data.xls")
    df = load_shiller("ie_data.xls", start="1881-01-01", end="2014-12-01")
    df = load_shiller("ie_data.xls", cols=["date", "cape", "real_price"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Column mapping: raw index → (clean_name, description)
# Columns 13 and 15 are empty in the source file and are dropped.
# ---------------------------------------------------------------------------
_COLUMNS: dict[int, tuple[str, str]] = {
    0:  ("date_shiller",               "Raw Shiller decimal date (e.g. 1871.01)"),
    1:  ("price",                      "S&P Composite nominal price index"),
    2:  ("dividend",                   "Nominal dividend per share (annualized)"),
    3:  ("earnings",                   "Nominal earnings per share (annualized)"),
    4:  ("cpi",                        "Consumer Price Index"),
    5:  ("date_fraction",              "Date as year fraction"),
    6:  ("gs10",                       "10-Year Treasury interest rate (GS10)"),
    7:  ("real_price",                 "Real S&P price index (CPI-adjusted)"),
    8:  ("real_dividend",              "Real dividend per share"),
    9:  ("real_tr_price",              "Real Total Return price (dividends reinvested)"),
    10: ("real_earnings",              "Real earnings per share"),
    11: ("real_tr_earnings",           "Real TR scaled earnings per share"),
    12: ("cape",                       "CAPE ratio – Shiller P/E10 (price index)"),
    # 13: empty column – dropped
    14: ("tr_cape",                    "TR CAPE ratio – P/E10 on Total Return index"),
    # 15: empty column – dropped
    16: ("excess_cape_yield",          "Excess CAPE Yield (1/CAPE minus real bond yield)"),
    17: ("monthly_tr_bond_return",     "Monthly Total Return bond (nominal)"),
    18: ("real_tr_bond_return",        "Real Total Return bond"),
    19: ("real_10y_stock_return",      "10-Year annualized real stock return (RETt)"),
    20: ("real_10y_bond_return",       "10-Year annualized real bond return"),
    21: ("real_10y_excess_return",     "10-Year annualized real excess return (stocks minus bonds)"),
}

_DROP_IDX  = {13, 15}                          # empty source columns
_KEEP_IDX  = [i for i in _COLUMNS if i not in _DROP_IDX]
_COL_NAMES = [_COLUMNS[i][0] for i in _KEEP_IDX]

_DATA_START_ROW = 8   # first row of actual data in the sheet (0-indexed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_shiller(
    filepath: str | Path = "ie_data.xls",
    cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load Shiller's ie_data.xls and return a clean DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Path to ie_data.xls.
    cols : list[str] | None
        Columns to return. None returns all columns.
        Call ``shiller_info()`` to see available column names.

    Returns
    -------
    pd.DataFrame
        - Column ``date``  : datetime64[ns], first day of each month.
        - All other columns: float64 (Shiller 'NA' → np.nan).
        - Sorted ascending by date, index reset to 0-based integer.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If unknown column names are requested.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # -- 1. Read raw data --------------------------------------------------
    raw = _read_raw(filepath)

    # -- 2. Parse datetime column -----------------------------------------
    raw = _parse_date(raw)

    # -- 3. Cast everything else to float (Shiller 'NA' → NaN) -----------
    numeric_cols = [c for c in raw.columns if c != "date"]
    raw[numeric_cols] = raw[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # -- 4. Drop helper column --------------------------------------------
    raw = raw.drop(columns=["date_shiller", "date_fraction"], errors="ignore")


    # -- 5. Column selection ----------------------------------------------
    if cols is not None:
        _validate_cols(cols, raw.columns.tolist())
        # Always keep date first
        ordered = ["date"] + [c for c in cols if c != "date"]
        raw = raw[[c for c in ordered if c in raw.columns]]

    return raw


def shiller_info() -> pd.DataFrame:
    """Return a DataFrame describing all available columns."""
    rows = [
        {"column": _COLUMNS[i][0], "description": _COLUMNS[i][1]}
        for i in _KEEP_IDX
        if _COLUMNS[i][0] not in ("date_shiller", "date_fraction")
    ]
    # Insert the constructed `date` column at position 0
    rows.insert(0, {"column": "date", "description": "Date (datetime64, first day of month)"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_raw(filepath: Path) -> pd.DataFrame:
    """Read the raw Data sheet, keep only mapped columns, name them."""
    try:
        import xlrd  # noqa: F401
    except ImportError as exc:
        raise ImportError("xlrd is required: pip install xlrd") from exc

    raw = pd.read_excel(
        filepath,
        sheet_name="Data",
        header=None,
        skiprows=_DATA_START_ROW,
        engine="xlrd",
    )
    # Keep only indexed columns that exist
    valid_idx = [i for i in _KEEP_IDX if i < raw.shape[1]]
    df = raw.iloc[:, valid_idx].copy()
    df.columns = [_COLUMNS[i][0] for i in valid_idx]

    # Drop rows where date_shiller is clearly not a number
    df = df[pd.to_numeric(df["date_shiller"], errors="coerce").notna()]
    return df.reset_index(drop=True)


def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Shiller decimal date (1871.01) to ISO string '1871-01-01'.

    Stored as str (object dtype) so Excel can display pre-1900 dates
    correctly. Excel's date system only supports dates from 1900-01-01;
    earlier dates stored as datetime64 appear as '1/0/1900'.
    """
    def _to_str(x: float) -> str | None:
        if pd.isna(x):
            return None
        year  = int(x)
        month = int(round((x - year) * 100))
        month = max(1, min(month, 12))          # clamp to valid month
        return f"{year:04d}-{month:02d}-01"

    decimal = pd.to_numeric(df["date_shiller"], errors="coerce")
    df.insert(0, "date", decimal.apply(_to_str).astype("object"))
    return df


def _validate_cols(requested: Sequence[str], available: list[str]) -> None:
    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(
            f"Unknown column(s): {sorted(unknown)}\n"
            f"Available: {sorted(available)}"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = "/Users/juniorjumbong/Desktop/personal-website/00_tds/02_Shiller_PE_Ratio_CAPE/datasets/ie_data.xls"

    print("=" * 60)
    print("AVAILABLE COLUMNS")
    print("=" * 60)
    print(shiller_info().to_string(index=False))

    print("\n" + "=" * 60)
    print("FULL LOAD")
    print("=" * 60)
    df = load_shiller(path)
    print(f"Shape : {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"\nHead:\n{df.head(3).to_string()}")
    print(f"\nTail:\n{df.tail(3).to_string()}")
    print(f"\nNaN per column:\n{df.isna().sum().to_string()}")

    print("\n" + "=" * 60)
    print("FILTERED EXAMPLE: CAPE + real_price (1881-2014)")

    # save to excel to check formatting (especially pre-1900 dates)
    output_path = Path("/Users/juniorjumbong/Desktop/personal-website/00_tds/02_Shiller_PE_Ratio_CAPE/datasets/shiller_with_date_in_string_format.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\nSaved full DataFrame to {output_path} for manual inspection.")