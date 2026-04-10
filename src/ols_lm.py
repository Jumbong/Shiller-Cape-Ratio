"""
ols_lm.py
=========
DRY function to run an OLS regression (similar to R's lm()) using statsmodels,
with Newey-West HAC standard errors for time series.

Returns a dict containing:
    - equation    : str  — formatted estimated equation
    - coefficients: dict — {name: coef}
    - t_stats     : dict — {name: t-stat (HAC)}
    - p_values    : dict — {name: p-value (HAC)}
    - r2          : float
    - r2_adj      : float
    - residuals   : pd.Series
    - fitted      : pd.Series
    - summary     : str  — full statsmodels summary

Usage
-----
    from ols_lm import lm
    import numpy as np

    result = lm(y=d["real_10y_stock_return"] * 100,
                X=np.log(d["cape"]),
                y_name="RET_t",
                x_names=["log(CAPE)"])

    print(result["equation"])
    print(result["t_stats"])
    print(result["residuals"].head())
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


def lm(
    y: pd.Series | np.ndarray,
    X: pd.Series | pd.DataFrame | np.ndarray,
    y_name:  str = "y",
    x_names: Sequence[str] | None = None,
    add_constant: bool = True,
    n_lags: int | None = None,
    X_forecast: pd.Series | pd.DataFrame | np.ndarray | None = None,
) -> dict:
    """
    OLS regression with Newey-West HAC standard errors (equivalent to R's lm).

    Parameters
    ----------
    y : pd.Series | np.ndarray
        Dependent variable.
    X : pd.Series | pd.DataFrame | np.ndarray
        Independent variable(s). If pd.Series, treated as a single regressor.
    y_name : str
        Name of the dependent variable (used in equation string).
    x_names : list[str] | None
        Names of the regressors. None → ['x1', 'x2', ...].
    add_constant : bool
        Add intercept (default True).
    n_lags : int | None
        Number of lags for Newey-West. None → automatic (int(4*(n/100)^(2/9))).
    X_forecast : pd.Series | pd.DataFrame | np.ndarray | None
        Optional new X values on which to generate out-of-sample forecasts.
        Must have the same columns/order as X (constant added automatically).
        None → forecast key is None in the returned dict.

    Returns
    -------
    dict with keys:
        equation     : str             Formatted estimated equation
        coefficients : dict[str, float]
        std_errors   : dict[str, float] HAC standard errors
        t_stats      : dict[str, float] HAC t-statistics
        p_values     : dict[str, float] HAC p-values
        r2           : float
        r2_adj       : float
        n_obs        : int
        residuals    : pd.Series
        fitted       : pd.Series
        summary      : str             Full statsmodels summary (HAC)
        forecast     : pd.Series | None  Out-of-sample predictions from X_forecast
    """
    # -- 1. Align and clean data -----------------------------------------
    y = pd.Series(y).reset_index(drop=True)

    if isinstance(X, pd.Series):
        X = X.to_frame()
    elif isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    else:
        X = X.copy().reset_index(drop=True)

    X = X.reset_index(drop=True)

    # Name regressors
    if x_names is not None:
        X.columns = list(x_names)
    elif X.columns.tolist() == list(range(X.shape[1])):
        X.columns = [f"x{i+1}" for i in range(X.shape[1])]

    # Drop rows where y or any X is NaN
    combined = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    y_clean  = combined["__y__"]
    X_clean  = combined.drop(columns="__y__")

    n = len(y_clean)

    # -- 2. Add constant -------------------------------------------------
    if add_constant:
        X_clean = sm.add_constant(X_clean, has_constant="add")
        X_clean = X_clean.rename(columns={"const": "Intercept"})

    # -- 3. Fit OLS ------------------------------------------------------
    model  = sm.OLS(y_clean, X_clean)
    result = model.fit()

    # -- 4. Newey-West HAC standard errors --------------------------------
    if n_lags is None:
        n_lags = int(4 * (n / 100) ** (2 / 9))   # Andrews (1991) rule

    hac    = result.get_robustcov_results(cov_type="HAC", maxlags=n_lags)
    names  = X_clean.columns.tolist()

    coefficients = dict(zip(names, hac.params))
    std_errors   = dict(zip(names, hac.bse))
    t_stats      = dict(zip(names, hac.tvalues))
    p_values     = dict(zip(names, hac.pvalues))

    # -- 5. Build equation string ----------------------------------------
    equation = _build_equation(y_name, coefficients, t_stats, result.rsquared)

    # -- 6. Residuals and fitted values ----------------------------------
    residuals = pd.Series(np.asarray(hac.resid), name="residuals")
    fitted    = pd.Series(np.asarray(hac.fittedvalues), name="fitted")

    # -- 7. Out-of-sample forecast (optional) ----------------------------
    forecast = None
    if X_forecast is not None:
        if isinstance(X_forecast, pd.Series):
            X_fc = X_forecast.to_frame()
        elif isinstance(X_forecast, np.ndarray):
            X_fc = pd.DataFrame(X_forecast)
        else:
            X_fc = X_forecast.copy().reset_index(drop=True)

        # Apply same column names as training X
        X_fc.columns = [c for c in X_clean.columns if c != "Intercept"]

        if add_constant:
            X_fc = sm.add_constant(X_fc, has_constant="add")
            X_fc = X_fc.rename(columns={"const": "Intercept"})
            # Ensure column order matches training
            X_fc = X_fc[X_clean.columns]

        forecast = pd.Series(
            hac.predict(X_fc),
            name="forecast",
        )

    return {
        "equation":     equation,
        "coefficients": coefficients,
        "std_errors":   std_errors,
        "t_stats":      t_stats,
        "p_values":     p_values,
        "r2":           result.rsquared,
        "r2_adj":       result.rsquared_adj,
        "n_obs":        n,
        "n_lags_hac":   n_lags,
        "residuals":    residuals,
        "fitted":       fitted,
        "forecast":     forecast,
        "summary":      hac.summary().as_text(),
    }


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _build_equation(
    y_name: str,
    coefficients: dict[str, float],
    t_stats: dict[str, float],
    r2: float,
) -> str:
    """Format the estimated equation as a readable string."""
    terms = []
    for i, (name, coef) in enumerate(coefficients.items()):
        if name == "Intercept":
            terms.append(f"{coef:.4f}")
        else:
            sign = "+" if coef >= 0 else "-"
            terms.append(f"{sign} {abs(coef):.4f}·{name}")

    equation = f"{y_name} = " + " ".join(terms) + f"  [R² = {r2:.3f}]"

    # Add t-stats line below
    t_line = " " * (len(y_name) + 3)
    for name, t in t_stats.items():
        label = f"t={t:.2f}"
        if name == "Intercept":
            t_line += label.ljust(10)
        else:
            t_line += f"        {label}"

    return equation + "\n" + t_line


# ---------------------------------------------------------------------------
# Standalone demo — reproduces Campbell & Shiller (1998) regression
# ---------------------------------------------------------------------------
