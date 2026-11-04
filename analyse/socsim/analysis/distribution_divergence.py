from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, entropy

from socsim.analysis.comparisons import _discretise_for_counts
from socsim.analysis.descriptives import BFI_DISTRIBUTION_VARS, CORE_DISTRIBUTION_VARS, resolve_var_column

DISTRIBUTION_VARS = CORE_DISTRIBUTION_VARS + BFI_DISTRIBUTION_VARS


def _numeric_series(df: pd.DataFrame, variable: str, dataset: str) -> pd.Series:
    col = resolve_var_column(df, variable, dataset)
    if not col:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _probability_vector(variable: str, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    disc, levels = _discretise_for_counts(variable, values)
    counts = disc.value_counts(dropna=True)
    freq = np.array([float(counts.get(level, 0.0)) for level in levels], dtype=float)
    total = freq.sum()
    probs = np.full_like(freq, np.nan) if total <= 0 else freq / total
    return freq, probs


def _cohens_d(human: pd.Series, sim: pd.Series) -> float:
    if len(human) < 2 or len(sim) < 2:
        return np.nan
    human_var = human.var(ddof=1)
    sim_var = sim.var(ddof=1)
    denom_df = len(human) + len(sim) - 2
    if denom_df <= 0:
        return np.nan
    pooled = np.sqrt((((len(human) - 1) * human_var) + ((len(sim) - 1) * sim_var)) / denom_df)
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return float((sim.mean() - human.mean()) / pooled)


def _chi2(freq_human: np.ndarray, freq_sim: np.ndarray) -> tuple[float, float]:
    if freq_human.sum() <= 0 or freq_sim.sum() <= 0:
        return np.nan, np.nan
    observed = np.vstack([freq_human, freq_sim])
    try:
        chi2, p_value, _, _ = chi2_contingency(observed)
    except ValueError:
        return np.nan, np.nan
    return float(chi2), float(p_value)


def compute_distribution_divergence(
    by_dataset: dict[str, pd.DataFrame],
    variables: list[str] | None = None,
) -> pd.DataFrame:
    variables = variables or DISTRIBUTION_VARS
    if "human" not in by_dataset:
        return pd.DataFrame(
            columns=[
                "variable",
                "comparison",
                "mean_human",
                "mean_sim",
                "mean_diff",
                "cohens_d",
                "js_divergence",
                "entropy_human",
                "entropy_sim",
                "entropy_diff",
                "chi2",
                "chi2_p",
            ]
        )

    comparison_datasets = [ds for ds in by_dataset if ds == "synthetic" or ds.endswith("_agent")]
    rows: list[dict[str, object]] = []
    for variable in variables:
        human_values = _numeric_series(by_dataset["human"], variable, "human")
        if human_values.empty:
            continue
        human_freq, human_probs = _probability_vector(variable, human_values)
        human_entropy = float(entropy(human_probs)) if np.isfinite(human_probs).all() else np.nan
        human_mean = float(human_values.mean())

        for comparison in comparison_datasets:
            sim_values = _numeric_series(by_dataset[comparison], variable, comparison)
            if sim_values.empty:
                continue
            sim_freq, sim_probs = _probability_vector(variable, sim_values)
            sim_entropy = float(entropy(sim_probs)) if np.isfinite(sim_probs).all() else np.nan
            js_div = (
                float(jensenshannon(human_probs, sim_probs))
                if np.isfinite(human_probs).all() and np.isfinite(sim_probs).all()
                else np.nan
            )
            chi2, chi2_p = _chi2(human_freq, sim_freq)
            sim_mean = float(sim_values.mean())
            rows.append(
                {
                    "variable": variable,
                    "comparison": comparison,
                    "mean_human": human_mean,
                    "mean_sim": sim_mean,
                    "mean_diff": sim_mean - human_mean,
                    "cohens_d": _cohens_d(human_values, sim_values),
                    "js_divergence": js_div,
                    "entropy_human": human_entropy,
                    "entropy_sim": sim_entropy,
                    "entropy_diff": sim_entropy - human_entropy,
                    "chi2": chi2,
                    "chi2_p": chi2_p,
                }
            )
    return pd.DataFrame(rows)
