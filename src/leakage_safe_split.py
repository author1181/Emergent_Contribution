"""
leakage_safe_split.py
=====================

Implements the global temporal-cutoff mode for the DCNAR pipeline, per
leakage_safe_dcnar_spec.md §2. Replaces the per-country last-N-years holdout
with a single global year threshold C, so that the model and ALL preprocessing
are blind to every country's data after year C.

This is the machinery that makes "as-of-time-C Jacobian" a legitimate
forecasting object rather than retrospective reconstruction.

The three leakage channels closed here:
  (b) estimation   — training rows are strictly year <= C - v
  (c) preprocessing — mu, sigma fit on training rows ONLY, then frozen
  (partition/threshold leakage handled by passing train-only frames downstream)

Predictor leakage (a) is closed in the early-warning predictor step, not here.

Design rule (frozen pre-registration §5): block-wise temporal holdout is the
PRIMARY design. If too few post-C episodes survive, the run is flagged
UNDERPOWERED — never silently swapped for a more favorable design.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CutoffSplit:
    """
    Output of build_cutoff_split. Holds the three disjoint frames plus the
    FROZEN preprocessing parameters and provenance metadata for the manifest.
    """
    cutoff_C: int
    inner_val_band: int

    train_df: pd.DataFrame          # year <= C - v   (model + preprocessing fit here)
    inner_val_df: pd.DataFrame      # C - v < year <= C   (checkpoint selection only)
    test_df: pd.DataFrame           # year > C   (held aside; never touches fitting)

    # Frozen preprocessing (fit on train_df ONLY)
    mu: np.ndarray
    sigma: np.ndarray
    standardization_fit_on: str     # provenance string, asserted downstream

    components: List[str]
    time_col: str
    unit_col: str

    # Provenance / manifest fields
    n_train_rows: int
    n_inner_val_rows: int
    n_test_rows: int
    n_train_units: int
    n_test_units: int

    def standardize(self, X: np.ndarray) -> np.ndarray:
        """Apply frozen (mu, sigma). Works on any rows, including post-C."""
        return (X - self.mu) / self.sigma

    def provenance(self) -> Dict:
        return {
            "cutoff_C": self.cutoff_C,
            "inner_val_band": self.inner_val_band,
            "standardization_fit_on": self.standardization_fit_on,
            "n_train_rows": self.n_train_rows,
            "n_inner_val_rows": self.n_inner_val_rows,
            "n_test_rows": self.n_test_rows,
            "n_train_units": self.n_train_units,
            "n_test_units": self.n_test_units,
        }


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_cutoff_split(
    df: pd.DataFrame,
    *,
    components: List[str],
    cutoff_C: int,
    time_col: str = "year",
    unit_col: str = "country_id",
    inner_val_band: int = 5,
    eps: float = 1e-12,
) -> CutoffSplit:
    """
    Build the leakage-safe global temporal split.

    Partitions:
        train      : year <= C - inner_val_band
        inner_val  : C - inner_val_band < year <= C
        test       : year > C

    Standardization (mu, sigma) is fit on TRAIN ONLY and frozen.

    All three leakage assertions (spec §7) are enforced here and will raise if
    violated. Leakage bugs are silent; these assertions make them loud.
    """
    if time_col not in df.columns:
        raise ValueError(f"time_col {time_col!r} not in dataframe")
    if unit_col not in df.columns:
        raise ValueError(f"unit_col {unit_col!r} not in dataframe")
    missing = [c for c in components if c not in df.columns]
    if missing:
        raise ValueError(f"components missing from dataframe: {missing}")

    v = inner_val_band
    df = df.sort_values([unit_col, time_col]).reset_index(drop=True)

    train_df     = df[df[time_col] <= cutoff_C - v].copy()
    inner_val_df = df[(df[time_col] > cutoff_C - v) & (df[time_col] <= cutoff_C)].copy()
    test_df      = df[df[time_col] > cutoff_C].copy()

    if len(train_df) == 0:
        raise ValueError(
            f"No training rows with year <= {cutoff_C - v}. "
            f"Cutoff C={cutoff_C} minus inner_val_band v={v} is too early "
            f"for this panel (earliest year {int(df[time_col].min())})."
        )

    # ----- FROZEN preprocessing: fit on TRAIN rows only -----
    Xtr = train_df[components].to_numpy(float)
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma = np.where(sigma < eps, 1.0, sigma)

    split = CutoffSplit(
        cutoff_C=cutoff_C,
        inner_val_band=v,
        train_df=train_df,
        inner_val_df=inner_val_df,
        test_df=test_df,
        mu=mu,
        sigma=sigma,
        standardization_fit_on="train_rows_le_C_minus_v",
        components=list(components),
        time_col=time_col,
        unit_col=unit_col,
        n_train_rows=len(train_df),
        n_inner_val_rows=len(inner_val_df),
        n_test_rows=len(test_df),
        n_train_units=train_df[unit_col].nunique(),
        n_test_units=test_df[unit_col].nunique(),
    )

    _assert_no_leakage(split)
    return split


# ---------------------------------------------------------------------------
# The assertions that matter most (spec §7)
# ---------------------------------------------------------------------------

def _assert_no_leakage(s: CutoffSplit) -> None:
    """
    Hard guards. These must never be removed. Each corresponds to a leakage
    channel; a violation means future information could reach the model.
    """
    tc = s.time_col

    # Test rows strictly after the cutoff
    if len(s.test_df):
        assert s.test_df[tc].min() > s.cutoff_C, (
            f"LEAKAGE: test rows include year <= C ({s.test_df[tc].min()} <= {s.cutoff_C})"
        )

    # Training rows strictly at or before C - v
    assert s.train_df[tc].max() <= s.cutoff_C - s.inner_val_band, (
        f"LEAKAGE: train rows include year > C - v "
        f"({s.train_df[tc].max()} > {s.cutoff_C - s.inner_val_band})"
    )

    # Inner-validation rows at or before C (never post-C)
    if len(s.inner_val_df):
        assert s.inner_val_df[tc].max() <= s.cutoff_C, (
            f"LEAKAGE: inner-val rows include year > C "
            f"({s.inner_val_df[tc].max()} > {s.cutoff_C})"
        )

    # No row index appears in more than one partition
    tr_idx = set(s.train_df.index)
    iv_idx = set(s.inner_val_df.index)
    te_idx = set(s.test_df.index)
    assert tr_idx.isdisjoint(iv_idx), "LEAKAGE: train and inner-val indices overlap"
    assert tr_idx.isdisjoint(te_idx), "LEAKAGE: train and test indices overlap"
    assert iv_idx.isdisjoint(te_idx), "LEAKAGE: inner-val and test indices overlap"

    # Standardization provenance is the train-only string
    assert s.standardization_fit_on == "train_rows_le_C_minus_v", (
        f"LEAKAGE: standardization provenance is {s.standardization_fit_on!r}, "
        "expected 'train_rows_le_C_minus_v'"
    )


# ---------------------------------------------------------------------------
# Power check (frozen pre-registration §5)
# ---------------------------------------------------------------------------

def assess_power(
    split: CutoffSplit,
    episode_onsets: pd.DataFrame,
    *,
    horizon_h: int = 1,
    pre_window: int = 5,
    adequacy_threshold: int = 25,
    onset_year_col: str = "onset_year",
    unit_col: Optional[str] = None,
) -> Dict:
    """
    Count eligible test episodes (onset > C + h) and decide the UNDERPOWERED
    flag. Per pre-registration: block-holdout stays primary; if too few
    episodes survive, flag UNDERPOWERED — do NOT switch design here.

    episode_onsets: DataFrame with at least [unit_col, onset_year_col].
    Returns a dict for the manifest. Does not alter the split or the design.
    """
    uc = unit_col or split.unit_col
    C = split.cutoff_C

    eligible = episode_onsets[episode_onsets[onset_year_col] > C + horizon_h]
    n_eligible = len(eligible)

    return {
        "n_eligible_test_episodes": int(n_eligible),
        "adequacy_threshold": adequacy_threshold,
        "underpowered": bool(n_eligible < adequacy_threshold),
        "cutoff_C": C,
        "horizon_h": horizon_h,
        "pre_window": pre_window,
        "note": (
            "Block-holdout is primary regardless. If underpowered=True, report "
            "the result as underpowered; expanding-window is robustness, not "
            "an ex-post replacement (pre-registration §5)."
        ),
    }
