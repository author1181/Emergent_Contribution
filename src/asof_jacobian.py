"""
asof_jacobian.py
================

The genuinely new step (leakage_safe_dcnar_spec.md §3, §4): evaluate the
FROZEN model's local Jacobian at OBSERVED post-C states, and compute the
three early-warning stability diagnostics from those Jacobians.

The operator:

    A_{c,t} = d f_hat_{<=C}(x_{c,t}) / dx ,   for t > C

The model is frozen at C; the states at which we differentiate it are real
observed post-C states. This is the as-of-C forecasting operator — legitimate
because deployment forecasting does exactly this (apply a past-trained model to
today's observed state).

Diagnostics (synthetic-validated battery, pre-registration §4):
  (1) rho        = |lambda_max(A)|                     spectral radius
  (2) lead_real  = Re(lambda_max(A))   -> +1            classical slowing-down
                   (also report lead_imag, and a bifurcation-type label, since
                    lambda -> -1 is period-doubling and complex -> unit circle
                    is Neimark-Sacker — DIFFERENT mechanisms)
  (3) sigma_k    = largest singular value of Phi^k,    non-normal transient
                   Phi^k = A_{t+k-1} ... A_t            amplification

This module is estimator-agnostic about HOW the Jacobian is produced; it offers
a torch autodiff path (Route A) and accepts any callable that returns A given a
state. Route B (NAVAR contributions) is intentionally NOT used as a Jacobian
substitute — see spec §4.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Diagnostics from a single Jacobian / a product of Jacobians
# ---------------------------------------------------------------------------

def spectral_radius(A: np.ndarray) -> float:
    """
    rho = max MODULUS eigenvalue. Governs asymptotic growth/decay magnitude.
    NOTE (review #6): rho is the largest |lambda|, which is NOT necessarily the
    same eigenvalue returned by leading_eigenvalue() (largest REAL part). On a
    non-normal or complex-spectrum system the eigenvalue driving rho and the one
    reported as lead_real can differ. Report both and do not conflate them.
    """
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def leading_eigenvalue(A: np.ndarray) -> complex:
    """
    Eigenvalue with the largest REAL part — the one relevant to classical
    critical slowing down (Re -> +1). NOTE (review #6): this is generally NOT
    the eigenvalue of largest modulus (see spectral_radius). lead_real and rho
    coincide only when the largest-real-part eigenvalue is also the largest in
    modulus (e.g. a real dominant eigenvalue). Reporting must keep them distinct.
    """
    ev = np.linalg.eigvals(A)
    return complex(ev[np.argmax(ev.real)])


def bifurcation_label(A: np.ndarray, tol: float = 0.15) -> str:
    """
    Classify the approaching-instability TYPE from the leading eigenvalue.
    Distinct mechanisms must not be lumped (pre-registration §4):
      - 'fold/transcritical'  : leading eig real and -> +1
      - 'period-doubling'     : leading eig real and -> -1
      - 'neimark-sacker'      : leading eig complex with modulus -> 1
      - 'stable'              : modulus comfortably < 1
    """
    ev = np.linalg.eigvals(A)
    lead = ev[np.argmax(np.abs(ev))]   # closest to unit circle by modulus
    mod = abs(lead)
    if mod < 1 - tol:
        return "stable"
    if abs(lead.imag) > tol:
        return "neimark-sacker"
    if lead.real > 0:
        return "fold/transcritical"
    return "period-doubling"


def finite_time_sigma(jacobians: List[np.ndarray]) -> float:
    """
    Largest singular value of the finite-time product Phi^k = A_{k-1} ... A_0.
    Catches non-normal transient amplification that eigenvalues miss.
    `jacobians` is the ordered list [A_t, A_{t+1}, ..., A_{t+k-1}].
    """
    n = jacobians[0].shape[0]
    Phi = np.eye(n)
    for A in jacobians:                 # product A_{t+k-1} ... A_t
        Phi = A @ Phi
    return float(np.linalg.svd(Phi, compute_uv=False)[0])


# ---------------------------------------------------------------------------
# Autodiff Jacobian (Route A) — torch
# ---------------------------------------------------------------------------

def make_torch_jacobian_fn(
    model,
    *,
    n_components: int,
    maxlags: int,
    lag_reduction: str = "companion",
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a function state_window -> A (n x n) using torch autodiff on a FROZEN
    model. Route A from spec §4.

    The model forward is assumed to be `preds, _ = model(x)` with x of shape
    (1, N, K) (batch, components, lags), matching the NAVAR conv1d convention in
    the notebook.

    lag_reduction:
      - 'lag1'      : d preds / d(most-recent lag) — the lag-1 block only (N x N).
                      A LOCAL one-step block, NOT the full multi-lag stability
                      operator. Adequate for a quick read; not rigorous for a
                      K-lag system.
      - 'sum'       : sum the Jacobian over all lag positions (N x N). A crude
                      collapse; NOT equal to the companion matrix and NOT a
                      correct stability object. Kept for backward compatibility
                      only; prefer 'companion'.
      - 'companion' : the (N*K) x (N*K) COMPANION Jacobian of the linearized
                      K-lag system (review #5). Top block-row holds the K blocks
                      [B_1 | B_2 | ... | B_K] where B_l = d preds / d(lag l), with
                      lag 1 = most recent; the sub-block-diagonal is identity
                      (the state shift). Its eigenvalues are the TRUE multipliers
                      of the K-lag system — this is the mathematically correct
                      object for spectral_radius / leading_eigenvalue / slowing-
                      down. Returns shape (N*K, N*K).

    Use ONE convention everywhere (E_j, diagnostics, IRF) and document it. For
    rigorous stability claims on a multi-lag model, use 'companion'.

    The model must already be frozen (eval mode, no grad on params). Only the
    input is differentiated.
    """
    import torch

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def jac_fn(state_window: np.ndarray) -> np.ndarray:
        # state_window: (N, K) already standardized with FROZEN mu/sigma
        x = torch.tensor(
            state_window.reshape(1, n_components, maxlags),
            dtype=torch.float32, device=device, requires_grad=True,
        )

        def g(inp):
            preds, _ = model(inp)
            return preds.reshape(-1)          # (N,)

        # full Jacobian: (N, 1, N, K)
        J = torch.autograd.functional.jacobian(g, x, create_graph=False)
        J = J.reshape(n_components, n_components, maxlags).cpu().numpy()
        # J[i, j, l] = d preds_i / d input_{j, lag-position l}
        # NOTE: window is OLD->NEW, so column l=-1 is the MOST RECENT lag (lag 1).

        N, K = n_components, maxlags
        if lag_reduction == "lag1":
            return J[:, :, -1]                # (N, N) most-recent block
        elif lag_reduction == "sum":
            return J.sum(axis=2)              # (N, N) crude collapse
        elif lag_reduction == "companion":
            # Build (N*K, N*K) companion. Block-lag ordering newest->oldest:
            # B_1 = lag 1 = J[:,:,-1], B_2 = J[:,:,-2], ..., B_K = J[:,:,0].
            C = np.zeros((N * K, N * K), dtype=float)
            for l in range(K):
                B_l = J[:, :, K - 1 - l]      # l=0 -> most recent (lag 1)
                C[0:N, l * N:(l + 1) * N] = B_l
            # sub-block-diagonal identity (state shift): block row r>=1 carries
            # I in block column r-1.
            for r in range(1, K):
                C[r * N:(r + 1) * N, (r - 1) * N:r * N] = np.eye(N)
            return C                          # (N*K, N*K)
        else:
            raise ValueError(f"unknown lag_reduction {lag_reduction!r}")

    return jac_fn


# ---------------------------------------------------------------------------
# Evaluate diagnostics over post-C country-years
# ---------------------------------------------------------------------------

@dataclass
class AsOfCResult:
    """Diagnostics keyed by (unit, year), plus the raw Jacobians."""
    records: list          # list of dicts: unit, year, rho, lead_real, lead_imag,
                           #                 bif_type, sigma_k
    jacobians: dict        # (unit, year) -> A (n x n)
    cutoff_C: int
    k_horizon: int


def evaluate_asof_c(
    split,                                  # CutoffSplit (frozen mu/sigma, test_df)
    jac_fn: Callable[[np.ndarray], np.ndarray],
    *,
    maxlags: int,
    k_horizon: int = 5,
) -> AsOfCResult:
    """
    For each test country-year t > C, build the OBSERVED lag window, standardize
    with the FROZEN (mu, sigma), compute A_{c,t} via jac_fn, and the three
    diagnostics. For sigma_k, the product runs over the next k observed states
    (all evaluated through the frozen model — leakage-safe).

    Requires the full per-unit series (including <= C history) to build lag
    windows at the first post-C years. We therefore read the lag context from
    the original (unstandardized) panel reconstructed from split frames.
    """
    comps = split.components
    tc, uc = split.time_col, split.unit_col

    # (review #7) HARD GUARD: the entire leakage-safety of this routine rests on
    # split.standardize() applying FROZEN pre-C means/SDs. If the split's
    # provenance is anything other than the frozen-train flag, refuse to run —
    # a re-fit-on-full-panel scaler would silently leak the future into every
    # post-C Jacobian.
    prov = getattr(split, "standardization_fit_on", None)
    assert prov == "train_rows_le_C_minus_v", (
        f"LEAKAGE GUARD: split.standardization_fit_on={prov!r}, expected "
        "'train_rows_le_C_minus_v'. as-of-C Jacobians require frozen pre-C "
        "standardization; refusing to proceed."
    )

    # Reconstruct full per-unit series (train + inner_val + test), sorted.
    full = (
        __import__("pandas")
        .concat([split.train_df, split.inner_val_df, split.test_df])
        .sort_values([uc, tc])
        .reset_index(drop=True)
    )

    records = []
    jac_store = {}

    for unit, g in full.groupby(uc, sort=False):
        g = g.sort_values(tc).reset_index(drop=True)
        years = g[tc].to_numpy()
        X = g[comps].to_numpy(float)
        Xz = split.standardize(X)            # frozen mu/sigma

        # Precompute the Jacobian at every time index where a full lag window
        # exists (need maxlags history). Cache so sigma_k can reuse them.
        A_by_idx: Dict[int, np.ndarray] = {}

        n_idx = len(g)

        def A_at(idx: int) -> Optional[np.ndarray]:
            # Need a full lag window [idx-maxlags, idx); idx must be a valid row.
            if idx < maxlags or idx >= n_idx:
                return None
            if idx not in A_by_idx:
                window = Xz[idx - maxlags: idx, :].T    # (N, K)
                A_by_idx[idx] = jac_fn(window)
            return A_by_idx[idx]

        for idx in range(len(g)):
            t = int(years[idx])
            if t <= split.cutoff_C:
                continue                      # only post-C years are test points
            A = A_at(idx)
            if A is None:
                continue                      # not enough lag history

            lead = leading_eigenvalue(A)
            rec = {
                uc: unit, tc: t,
                "rho": spectral_radius(A),
                "lead_real": float(lead.real),
                "lead_imag": float(lead.imag),
                "bif_type": bifurcation_label(A),
            }

            # sigma_k over the next k observed states (if available)
            seq = []
            ok = True
            for s in range(k_horizon):
                A_s = A_at(idx + s)
                if A_s is None:
                    ok = False
                    break
                seq.append(A_s)
            rec["sigma_k"] = finite_time_sigma(seq) if ok else float("nan")

            records.append(rec)
            jac_store[(unit, t)] = A

    return AsOfCResult(
        records=records,
        jacobians=jac_store,
        cutoff_C=split.cutoff_C,
        k_horizon=k_horizon,
    )
