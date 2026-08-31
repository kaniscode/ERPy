from __future__ import annotations

import itertools
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def compare_groups(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    groups: Iterable[str] | None = None,
    test: str = "welch",
    correction: str | None = "fdr_bh",
) -> pd.DataFrame:
    groups = list(groups or df[group_col].dropna().unique())
    rows = []
    for a, b in itertools.combinations(groups, 2):
        xa = pd.to_numeric(df.loc[df[group_col] == a, value_col], errors="coerce").dropna()
        xb = pd.to_numeric(df.loc[df[group_col] == b, value_col], errors="coerce").dropna()
        if test == "welch":
            stat, p = stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit")
        elif test in {"mannwhitney", "mann-whitney", "mw"}:
            stat, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        else:
            raise ValueError("test must be 'welch' or 'mannwhitney'")
        diff = float(xa.mean() - xb.mean())
        pooled = np.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2.0)
        rows.append(
            {
                "group_a": a,
                "group_b": b,
                "n_a": len(xa),
                "n_b": len(xb),
                "mean_a": xa.mean(),
                "mean_b": xb.mean(),
                "mean_diff": diff,
                "cohens_d": diff / pooled if pooled else np.nan,
                "statistic": stat,
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    if correction and not out.empty:
        try:
            from statsmodels.stats.multitest import multipletests

            reject, p_corr, _, _ = multipletests(out["p_value"].fillna(1), method=correction)
            out["p_corrected"] = p_corr
            out["reject"] = reject
            out["correction"] = correction
        except Exception:
            out["p_corrected"] = out["p_value"]
            out["reject"] = out["p_value"] < 0.05
            out["correction"] = "none"
    return out


def forest_plot(table: pd.DataFrame, effect_col: str = "cohens_d", label_cols=("group_a", "group_b")):
    labels = table[list(label_cols)].astype(str).agg(" vs ".join, axis=1)
    effects = pd.to_numeric(table[effect_col], errors="coerce")
    fig, ax = plt.subplots(figsize=(7, max(2, 0.35 * len(table))))
    y = np.arange(len(table))
    ax.axvline(0, color="0.4", lw=1)
    ax.scatter(effects, y, color="#2563eb")
    ax.set_yticks(y, labels)
    ax.set_xlabel(effect_col)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax
