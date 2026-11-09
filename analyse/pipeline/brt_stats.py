"""Estimators for the BRT outcome models: Wilson intervals, ML logistic regression with an automatic switch
to Firth-penalised logistic regression (rare events or separation), the simulated minimum detectable odds
ratio, Benjamini-Hochberg q values, and the agent-side prevalence-gap and association-transfer quantities.

Pure statistics: no paths, no I/O.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

Z95 = stats.norm.ppf(0.975)
CHI2_1_95 = stats.chi2.ppf(0.95, 1)


# ---------------------------------------------------------------------------- proportions

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((c - h) / d, (c + h) / d)


# ---------------------------------------------------------------------------- ML logistic (IRLS)

def _irls(X: np.ndarray, y: np.ndarray, maxiter: int = 60, tol: float = 1e-10):
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    sep = False
    for _ in range(maxiter):
        eta = X @ beta
        eta = np.clip(eta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        A = XtW @ X
        try:
            upd = np.linalg.solve(A, XtW @ z)
        except np.linalg.LinAlgError:
            sep = True
            break
        if not np.all(np.isfinite(upd)):
            sep = True
            break
        if np.max(np.abs(upd - beta)) < tol:
            beta = upd
            break
        beta = upd
    big = np.where(np.abs(beta) > 15)[0]
    if len(big):
        sep = True
    eta = np.clip(X @ beta, -35, 35)
    mu = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * np.log(np.clip(mu, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1))))
    w = np.clip(mu * (1 - mu), 1e-10, None)
    cov = np.linalg.pinv((X.T * w) @ X)
    return beta, ll, cov, (sep, big.tolist() if len(big) else [])


# ---------------------------------------------------------------------------- Firth logistic (Firth 1993)

def _firth_fit(X: np.ndarray, y: np.ndarray, maxiter: int = 200, tol: float = 1e-9,
               fixed: dict[int, float] | None = None):
    """Penalised-likelihood fit. `fixed` pins coefficients (for profile intervals and the LR test)."""
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    free = [j for j in range(p) if not (fixed and j in (fixed or {}))]
    if fixed:
        for j, v in fixed.items():
            beta[j] = v
    step = 1.0
    for _ in range(maxiter):
        eta = np.clip(X @ beta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        XtW = X.T * w
        I = XtW @ X
        try:
            Iinv = np.linalg.inv(I)
        except np.linalg.LinAlgError:
            Iinv = np.linalg.pinv(I)
        # hat diagonal of W^{1/2} X I^{-1} X' W^{1/2}
        sw = np.sqrt(w)
        Xs = X * sw[:, None]
        h = np.einsum("ij,jk,ik->i", Xs, Iinv, Xs)
        U = X.T @ (y - mu + h * (0.5 - mu))
        if not free:
            break
        Isub = I[np.ix_(free, free)]
        try:
            delta = np.linalg.solve(Isub, U[free])
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(Isub) @ U[free]
        delta = np.clip(delta, -5, 5)
        trial = beta.copy()
        trial[free] = beta[free] + step * delta
        if np.max(np.abs(trial - beta)) < tol:
            beta = trial
            break
        beta = trial
    eta = np.clip(X @ beta, -35, 35)
    mu = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * np.log(np.clip(mu, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - mu, 1e-12, 1))))
    w = np.clip(mu * (1 - mu), 1e-10, None)
    I = (X.T * w) @ X
    sign, logdet = np.linalg.slogdet(I)
    pll = ll + 0.5 * logdet if sign > 0 else ll
    cov = np.linalg.pinv(I)
    return beta, pll, cov


def firth_logit(X: np.ndarray, y: np.ndarray, col: int = 1):
    """Firth fit, penalised-LR p and profile-likelihood 95% interval for coefficient `col`."""
    beta, pll_full, cov = _firth_fit(X, y)
    _, pll_null, _ = _firth_fit(X, y, fixed={col: 0.0})
    lr = 2 * (pll_full - pll_null)
    p = float(stats.chi2.sf(max(lr, 0.0), 1))
    target = pll_full - CHI2_1_95 / 2

    def g(b):
        _, pll, _ = _firth_fit(X, y, fixed={col: b})
        return pll - target

    se = float(np.sqrt(max(cov[col, col], 1e-12)))
    b0 = float(beta[col])
    lo = hi = np.nan
    for direction, name in ((-1, "lo"), (1, "hi")):
        step = max(se, 0.05)
        a, b = b0, b0 + direction * step
        for _ in range(60):
            if g(b) < 0:
                break
            a, b = b, b + direction * step
            step *= 1.6
            if abs(b - b0) > 60:
                break
        else:
            continue
        if g(b) >= 0:
            continue
        try:
            root = optimize.brentq(g, a, b, xtol=1e-6, maxiter=200)
        except ValueError:
            continue
        if name == "lo":
            lo = root
        else:
            hi = root
    return {"beta": b0, "se": se, "p": p, "ci_lo": lo, "ci_hi": hi, "lr": lr}


# ---------------------------------------------------------------------------- the reported binary model

def fit_binary(y: pd.Series, Xdf: pd.DataFrame, focal: str, force_firth: bool = False,
               event_floor: int = 20):
    """ML logistic regression with the Firth switch (fewer than `event_floor` events or non-events, or
    separation). Returns one dict; `focal` names the reported column. The p value is a likelihood-ratio p."""
    dat = pd.concat([y.rename("__y"), Xdf], axis=1).dropna()
    yv = dat["__y"].to_numpy(dtype=float)
    names = list(Xdf.columns)
    X = np.column_stack([np.ones(len(dat))] + [dat[c].to_numpy(dtype=float) for c in names])
    col = 1 + names.index(focal)
    events = int(yv.sum())
    nonevents = int(len(yv) - events)
    rare = min(events, nonevents) < event_floor

    beta, ll1, cov, (sep, sep_cols) = _irls(X, yv)
    use_firth = bool(force_firth or rare or sep)
    flags = []
    if rare:
        flags.append(f"events<{event_floor}")
    if sep:
        term_names = ["(intercept)"] + names
        bad = ", ".join(term_names[j] for j in sep_cols if j < len(term_names))
        flags.append(f"separation/non-convergence on: {bad}" if bad else "separation/non-convergence")

    if use_firth:
        f = firth_logit(X, yv, col=col)
        est = {
            "estimator": "Firth logit",
            "beta": f["beta"], "se": f["se"],
            "or": float(np.exp(f["beta"])),
            "or_lo": float(np.exp(f["ci_lo"])) if np.isfinite(f["ci_lo"]) else np.nan,
            "or_hi": float(np.exp(f["ci_hi"])) if np.isfinite(f["ci_hi"]) else np.nan,
            "p": f["p"], "ci_type": "profile likelihood",
        }
        bfull, _, _ = _firth_fit(X, yv)
        ame = _ame(X, bfull, col)
    else:
        _, ll0, _, _ = _irls(np.delete(X, col, axis=1), yv)
        lr = 2 * (ll1 - ll0)
        se = float(np.sqrt(max(cov[col, col], 1e-12)))
        est = {
            "estimator": "ML logit",
            "beta": float(beta[col]), "se": se,
            "or": float(np.exp(beta[col])),
            "or_lo": float(np.exp(beta[col] - Z95 * se)),
            "or_hi": float(np.exp(beta[col] + Z95 * se)),
            "p": float(stats.chi2.sf(max(lr, 0.0), 1)), "ci_type": "Wald",
        }
        ame = _ame(X, beta, col)

    est.update({
        "n": int(len(dat)), "events": events, "non_events": nonevents,
        "prevalence": events / len(dat) if len(dat) else np.nan,
        "ame_pp": ame, "flags": "; ".join(flags), "focal": focal,
        "terms": ", ".join(names),
    })
    return est


def _ame(X: np.ndarray, beta: np.ndarray, col: int) -> float:
    """Average marginal effect of a one-unit change in column `col`, in percentage points."""
    eta = np.clip(X @ beta, -35, 35)
    mu = 1.0 / (1.0 + np.exp(-eta))
    return float(np.mean(mu * (1 - mu) * beta[col]) * 100)


# ---------------------------------------------------------------------------- minimum detectable OR

def _power_binary(x: np.ndarray, base_prev: float, log_or: float, nsim: int, rng) -> float:
    """Power of the ML LR test at `log_or`; the intercept is solved so the marginal prevalence is base_prev."""
    def marg(a):
        return np.mean(1.0 / (1.0 + np.exp(-(a + log_or * x)))) - base_prev
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if marg(mid) < 0:
            lo = mid
        else:
            hi = mid
    a = (lo + hi) / 2
    pi = 1.0 / (1.0 + np.exp(-(a + log_or * x)))
    X = np.column_stack([np.ones(len(x)), x])
    X0 = np.ones((len(x), 1))
    hits = 0
    for _ in range(nsim):
        y = (rng.random(len(x)) < pi).astype(float)
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue
        _, ll1, _, _ = _irls(X, y)
        _, ll0, _, _ = _irls(X0, y)
        if stats.chi2.sf(2 * (ll1 - ll0), 1) < 0.05:
            hits += 1
    return hits / nsim


def mdor_binary(x: np.ndarray, base_prev: float, nsim: int = 1000, target: float = 0.80,
                seed: int = 42, direction: str = "above"):
    """Smallest OR detectable with `target` power at the observed N, prevalence and predictor distribution
    (alpha .05 two-sided). direction 'above' -> OR > 1, 'below' -> OR < 1 (returned as exp(|log OR|))."""
    rng = np.random.default_rng(seed)
    sgn = 1.0 if direction == "above" else -1.0
    lo, hi = 0.0, sgn * np.log(40.0)
    if _power_binary(x, base_prev, hi, nsim, rng) < target:
        return np.nan
    for _ in range(14):
        mid = (lo + hi) / 2
        if _power_binary(x, base_prev, mid, nsim, rng) >= target:
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi))


# ---------------------------------------------------------------------------- multiplicity

def bh(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    if ok.sum():
        out[ok] = multipletests(p[ok], alpha=0.05, method="fdr_bh")[1]
    return out


# ---------------------------------------------------------------------------- agent-side quantities

def prevalence_gap_bootstrap(run_y: np.ndarray, human_y: np.ndarray, n_boot: int = 2000,
                             seed: int = 42) -> tuple[float, float, float]:
    """F1: run prevalence - human prevalence in pp, with a percentile bootstrap 95% interval; the two
    samples are resampled independently at their own N."""
    run_y = np.asarray(run_y, dtype=float)
    human_y = np.asarray(human_y, dtype=float)
    run_y = run_y[np.isfinite(run_y)]
    human_y = human_y[np.isfinite(human_y)]
    if len(run_y) == 0 or len(human_y) == 0:
        return (np.nan, np.nan, np.nan)
    gap = 100.0 * (run_y.mean() - human_y.mean())
    rng = np.random.default_rng(seed)
    nr, nh = len(run_y), len(human_y)
    draws = 100.0 * (
        run_y[rng.integers(0, nr, (n_boot, nr))].mean(axis=1)
        - human_y[rng.integers(0, nh, (n_boot, nh))].mean(axis=1)
    )
    return (float(gap), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def compressed_flag(gap_pp: float, h_lo: float, h_hi: float, r_lo: float, r_hi: float,
                    threshold_pp: float = 20.0) -> bool:
    """F1 'compressed': |gap| >= 20 pp and the human and run 95% Wilson intervals (proportions) do not overlap."""
    vals = [gap_pp, h_lo, h_hi, r_lo, r_hi]
    if not all(np.isfinite(v) for v in vals):
        return False
    no_overlap = (r_hi < h_lo) or (r_lo > h_hi)
    return bool(abs(gap_pp) >= threshold_pp and no_overlap)


def unstable_flag(events: float, n: float, floor: int = 20) -> bool:
    """A run's outcome with fewer than `floor` events or non-events."""
    if not all(np.isfinite(v) for v in (events, n)):
        return False
    return bool(min(events, n - events) < floor)


def transfer_verdict(b_run: float, se_run: float, b_hum: float, se_hum: float, unstable: bool) -> dict:
    """F3 on the run's own bCRT score.

    'transfers'         = the difference interval includes 0 and the run's own interval excludes the null in
                          the human direction
    'fails to transfer' = the difference interval excludes 0
    'indeterminate'     = otherwise
    'unstable'          = the run's outcome is unstable; reported, not counted in the transfer tally.
    """
    out = {"verdict": "unavailable", "diff": np.nan, "diff_lo": np.nan, "diff_hi": np.nan}
    if not all(np.isfinite(v) for v in (b_run, se_run, b_hum, se_hum)):
        return out
    z95 = 1.959963984540054
    diff = b_run - b_hum
    se = math.sqrt(se_run ** 2 + se_hum ** 2)
    lo, hi = diff - z95 * se, diff + z95 * se
    out.update({"diff": float(diff), "diff_lo": float(lo), "diff_hi": float(hi)})
    if unstable:
        out["verdict"] = "unstable"
        return out
    run_lo, run_hi = b_run - z95 * se_run, b_run + z95 * se_run
    same_direction = (run_lo > 0) if b_hum > 0 else ((run_hi < 0) if b_hum < 0 else False)
    if lo > 0 or hi < 0:
        out["verdict"] = "fails to transfer"
    elif same_direction:
        out["verdict"] = "transfers"
    else:
        out["verdict"] = "indeterminate"
    return out
