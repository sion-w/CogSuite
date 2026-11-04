from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from socsim.analysis.descriptives import KEY_VARS, resolve_var_column


def _build_work(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    cols = {}
    for var in KEY_VARS:
        c = resolve_var_column(df, var, dataset)
        if c is not None:
            cols[var] = pd.to_numeric(df[c], errors="coerce")
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def build_corr_matrix(df: pd.DataFrame, dataset: str, method: str = "spearman") -> pd.DataFrame:
    work = _build_work(df, dataset)
    if work.empty:
        return pd.DataFrame()
    return work.corr(method=method, min_periods=3)


def build_corr_p_matrix(
    df: pd.DataFrame,
    dataset: str,
    method: str = "spearman",
    fdr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = _build_work(df, dataset)
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    pmat = pd.DataFrame(index=work.columns, columns=work.columns, dtype=float)
    nmat = pd.DataFrame(index=work.columns, columns=work.columns, dtype=float)
    for a in work.columns:
        for b in work.columns:
            pair = work[[a, b]].dropna()
            nmat.loc[a, b] = float(len(pair))
            if len(pair) < 3:
                pmat.loc[a, b] = float("nan")
                continue
            if a == b:
                pmat.loc[a, b] = 0.0
                continue
            try:
                if method == "spearman":
                    _, p = stats.spearmanr(pair[a], pair[b], nan_policy="omit")
                else:
                    _, p = stats.pearsonr(pair[a], pair[b])
            except Exception:
                p = float("nan")
            pmat.loc[a, b] = float(p)
    if not fdr:
        return pmat, nmat
    corrected = pmat.copy()
    mask = ~np.eye(len(corrected), dtype=bool)
    tri_mask = np.triu(mask, k=1)
    pvals = corrected.values[tri_mask]
    valid = np.isfinite(pvals)
    if valid.any():
        _, qvals, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        pvals_corr = pvals.copy()
        pvals_corr[valid] = qvals
        corrected_values = corrected.values
        corrected_values[tri_mask] = pvals_corr
        corrected_values[(tri_mask.T)] = pvals_corr
        np.fill_diagonal(corrected_values, 0.0)
        corrected = pd.DataFrame(corrected_values, index=corrected.index, columns=corrected.columns)
    return corrected, nmat


def corr_diff_vs_human(corr_human: pd.DataFrame, corr_other: pd.DataFrame) -> pd.DataFrame:
    if corr_human.empty or corr_other.empty:
        return pd.DataFrame()
    common = [c for c in corr_human.columns if c in corr_other.columns]
    return corr_other.loc[common, common] - corr_human.loc[common, common]
