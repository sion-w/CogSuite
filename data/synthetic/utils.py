
# src/utils.py
from __future__ import annotations
import re, math, json, yaml
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd

def load_cfg(cfg_path: str | Path = "config/config.yaml") -> dict:
    return yaml.safe_load(open(cfg_path))


CRT_ITEM_TEXT_COLS = [f"{p}_q{i}" for p in ("crt2", "bcrt", "dcrt") for i in range(1, 5)]


def read_csv_keep_text(path: str | Path, text_cols=CRT_ITEM_TEXT_COLS, **kwargs) -> pd.DataFrame:
    """``pd.read_csv``, except that the named free-text answer columns keep literal answers.

    CR-23: pandas' default NA parsing turns the answer text "None", "NA", "N/A", "null" and
    "nan" into missing. "None" is the correct answer to CRT2 item 4 (two participants' answers
    were lost this way before scoring), so in answer columns only an empty cell is missing.
    Every other column is parsed exactly as ``pd.read_csv(path, **kwargs)`` would parse it.
    """
    df = pd.read_csv(path, **kwargs)
    present = [c for c in text_cols if c in df.columns]
    if present:
        text = pd.read_csv(path, usecols=present, dtype=str, keep_default_na=False, na_values=[""])
        for col in present:
            df[col] = text[col].to_numpy()
    return df

def qualtrics_read_twolevel(csv_path: str | Path) -> pd.DataFrame:
    """Read a standard 3-row-header Qualtrics CSV per DATA_CLEANING_PROTOCOL.md.

    Row 0 = short column names (used as columns), row 1 = full question text,
    row 2 = ImportId metadata. We keep row 0 as the header and drop rows 1 and 2
    via skiprows=[1, 2]. The previous header=[0, 1] + iloc[2:] approach silently
    dropped the first real respondent (the row immediately after the ImportId row),
    which is incorrect; this matches the protocol's `pd.read_csv(path, skiprows=[1, 2])`.
    """
    # CR-23: keep_default_na=False so a typed answer such as "None" stays text; only an empty
    # cell is missing (every column is read as str here, so nothing else changes).
    df = pd.read_csv(csv_path, skiprows=[1, 2], dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [str(c).strip() for c in df.columns]
    return df

def standardise_likert(series: pd.Series) -> pd.Series:
    """Map common Likert labels to numeric 1..5 (or 1..7 if present).

    Robust to formats like "4 = Somewhat Characteristic" by extracting a
    leading digit when present; otherwise falls back to label mapping.
    """
    s = series.astype(str).str.strip()

    # 1) If the string starts with a digit 1..7, use it directly
    leading_num = s.str.extract(r"^\s*([1-7])", expand=False)
    out = pd.to_numeric(leading_num, errors="coerce")

    # 2) Fallback: map common textual labels (case-insensitive)
    map5 = {
        "extremely uncharacteristic": 1,
        "somewhat uncharacteristic": 2,
        "uncharacteristic": 2,
        "uncertain": 3,
        "neutral": 3,
        "somewhat characteristic": 4,
        "characteristic": 4,
        "extremely characteristic": 5,
        "strongly disagree": 1,
        "disagree": 2,
        "neither agree nor disagree": 3,
        "agree": 4,
        "strongly agree": 5,
        "extremely characteristic of me": 5,
        "extremely uncharacteristic of me": 1,
    }
    fallback = s.str.lower().map(map5)
    out = out.fillna(fallback)

    return out.astype("float")

def reverse_score(x: pd.Series, max_score: int = 5, min_score: int = 1) -> pd.Series:
    return (max_score + min_score) - x

def cronbach_alpha(items_df: pd.DataFrame) -> float:
    x = items_df.apply(pd.to_numeric, errors='coerce')
    x = x.dropna(how="all", axis=0)
    k = x.shape[1]
    if k < 2: return float("nan")
    var_sum = x.var(ddof=1, axis=0).sum()
    var_total = x.sum(axis=1).var(ddof=1)
    if var_total <= 0: return float("nan")
    return float(k/(k-1) * (1 - var_sum/var_total))

def split_half_reliability(items_df: pd.DataFrame, n_splits: int = 500, random_state: int | None = 42) -> dict:
    """Estimate split-half reliability with Spearman–Brown correction.

    Parameters
    ----------
    items_df : pd.DataFrame
        Columns are item scores (ideally binary 0/1). Rows are respondents.
    n_splits : int
        Number of random splits to average over.
    random_state : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict
        {"mean": float, "sd": float, "median": float, "n_items": int, "n_obs": int}
    """
    x = items_df.apply(pd.to_numeric, errors='coerce')
    x = x.dropna(how="all", axis=0)
    n_items = x.shape[1]
    n_obs = x.shape[0]
    if n_items < 2 or n_obs < 3:
        return {"mean": float("nan"), "sd": float("nan"), "median": float("nan"), "n_items": int(n_items), "n_obs": int(n_obs)}
    rng = np.random.default_rng(random_state)
    sb_vals = []
    cols = list(x.columns)
    for _ in range(max(1, n_splits)):
        rng.shuffle(cols)
        a_cols = cols[: n_items//2]
        b_cols = cols[n_items//2 :]
        # If odd, distribute roughly equally
        if len(a_cols) == 0 or len(b_cols) == 0:
            continue
        a = x[a_cols].sum(axis=1)
        b = x[b_cols].sum(axis=1)
        valid = pd.concat([a,b], axis=1).dropna()
        if valid.shape[0] < 3:
            continue
        r = valid.iloc[:,0].corr(valid.iloc[:,1])
        if pd.isna(r):
            continue
        # Spearman–Brown prophecy formula to approximate full-test reliability
        sb = 2*r / (1 + r) if r < 1 else 1.0
        sb_vals.append(sb)
    if not sb_vals:
        return {"mean": float("nan"), "sd": float("nan"), "median": float("nan"), "n_items": int(n_items), "n_obs": int(n_obs)}
    arr = np.array(sb_vals, dtype=float)
    return {
        "mean": float(np.nanmean(arr)),
        "sd": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.nanmedian(arr)),
        "n_items": int(n_items),
        "n_obs": int(n_obs),
    }

def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float,float]:
    if n <= 3 or pd.isna(r): return (float("nan"), float("nan"))
    z = 0.5 * np.log((1+r)/(1-r))
    se = 1/ math.sqrt(n-3)
    zcrit = 1.96 if alpha==0.05 else abs(np.quantile(np.random.normal(size=100000), 1-alpha/2))
    lo = np.tanh(z - zcrit*se)
    hi = np.tanh(z + zcrit*se)
    return (float(lo), float(hi))

def corr_with_ci(x: pd.Series, y: pd.Series):
    valid = pd.concat([x,y], axis=1).dropna()
    if valid.shape[0] < 3:
        return {"r": float("nan"), "p": float("nan"), "n": int(valid.shape[0]), "ci_low": float("nan"), "ci_high": float("nan")}
    r = valid.iloc[:,0].corr(valid.iloc[:,1])
    # p-value (t-test)
    n = valid.shape[0]
    t = r * math.sqrt((n-2)/(1-r**2)) if abs(r) < 1 else float("inf")
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(abs(t), df=n-2))
    lo, hi = fisher_ci(r, n)
    return {"r": float(r), "p": float(p), "n": int(n), "ci_low": float(lo), "ci_high": float(hi)}

def cohen_d_two_groups(x: pd.Series, g: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "g": g}).dropna()
    if df["g"].nunique() != 2: return float("nan")
    gvals = df["g"].unique().tolist()
    a = df[df["g"]==gvals[0]]["x"]
    b = df[df["g"]==gvals[1]]["x"]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float("nan")
    sa2 = a.var(ddof=1); sb2 = b.var(ddof=1)
    sp = math.sqrt(((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2))
    return float((a.mean() - b.mean())/sp)
