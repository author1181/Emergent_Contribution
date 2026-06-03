"""
verifier_core.py — canonical CHECKS list + runner for the ICDM submission.

Mirrors the structure of the author's prior reproducibility verifiers: each paper
claim is mapped to a canonical source value and a tolerance, and the runner asserts
every one. This file is the AUDIT TRAIL — open it to see exactly what each check
does and where each number comes from.

Canonical sources (placed in the bundle, produced by the reproducibility notebook):
  phase_diagram_results.csv  — 20-seed phase grid
                               (struct,nl,pers,eps,ej_t,acr_t,acp_t,ej_acr,
                                d_real,d_pool,d_real_lo,d_real_hi,d_pool_lo,
                                d_pool_hi,wilcoxon_p,n,degenerate_control)
  deepdomain_ensemble.json   — per-node E_j mean/sd/rank-range, mean pairwise
                               Spearman, baseline correlations, val_loss, dims
  domain_placement.csv       — five-domain realized/pooled departure

Run:  python verify_paper_numbers.py [--root DIR] [--verbose]
"""
import numpy as np, pandas as pd, json, os

# --- tolerance conventions (match the author's prior practice) ---
T_EXACT = 0.001    # paper precise to 3 decimals
T_2DP   = 0.01     # "≈ X" rounded to 2dp
T_1DP   = 0.05     # "≈ X" rounded to 1dp
T_AGG   = 0.005    # cross-cell aggregates

def _phase(df, struct, nl, eps_set, col):
    """Mean of a phase-grid column over given structure/nonlinearity and an eps set."""
    s = df[(df.struct == struct) & (df.nl == nl) & (df.eps.isin(eps_set))]
    return float(s[col].mean())

# CHECKS: (claim_id, description, accessor, expected, tol)
# accessor is a lambda taking the loaded sources dict -> computed value.
# tol == 0 means an exact (==) match (used for integer ranks / counts).
def build_checks():
    NON_EX = ['small', 'med', 'large']
    return [
      # ---------- Table 1 / four-regimes: Delta_pooled (non-extreme) ----------
      ("T1.signflip.med",  "Table1 sign-reversal med Delta_pooled = 0.47",
         lambda S: _phase(S['phase'], 'sign-flip-switch', 'med', NON_EX, 'd_pool'), 0.47, T_2DP),
      ("T1.signflip.high", "Table1 sign-reversal high Delta_pooled = 0.58",
         lambda S: _phase(S['phase'], 'sign-flip-switch', 'high', NON_EX, 'd_pool'), 0.58, T_2DP),
      ("T1.signflip.low",  "Table1 sign-reversal low Delta_pooled = 0.26",
         lambda S: _phase(S['phase'], 'sign-flip-switch', 'low', NON_EX, 'd_pool'), 0.26, T_2DP),
      ("T1.reroute.high",  "Table1 persistent-reroute high Delta_pooled = 0.10",
         lambda S: _phase(S['phase'], 'persistent-switch', 'high', NON_EX, 'd_pool'), 0.10, T_2DP),
      ("T1.static.med",    "Table1 static med Delta_pooled = 0.05",
         lambda S: _phase(S['phase'], 'static', 'med', NON_EX, 'd_pool'), 0.05, T_2DP),
      ("T1.drift.med",     "Table1 smooth-drift med Delta_pooled approx -0.01",
         lambda S: _phase(S['phase'], 'smooth-drift', 'med', NON_EX, 'd_pool'), -0.01, T_2DP),
      # ---------- headline CIs (sign-flip) ----------
      ("CI.med.lo",  "sign-flip med CI lower = 0.32",
         lambda S: _phase(S['phase'], 'sign-flip-switch', 'med', NON_EX, 'd_pool_lo'), 0.32, T_2DP),
      ("CI.med.hi",  "sign-flip med CI upper = 0.63",
         lambda S: _phase(S['phase'], 'sign-flip-switch', 'med', NON_EX, 'd_pool_hi'), 0.63, T_2DP),
      ("CI.high.lo", "sign-flip high CI lower = 0.40",
         lambda S: _phase(S['phase'], 'sign-flip-switch', 'high', NON_EX, 'd_pool_lo'), 0.40, T_2DP),
      ("CI.high.hi", "sign-flip high CI upper = 0.77",
         lambda S: _phase(S['phase'], 'sign-flip-switch', 'high', NON_EX, 'd_pool_hi'), 0.77, T_2DP),
      # ---------- Wilcoxon seed-robustness fractions ----------
      ("W.signflip", "97% of sign-reversal cells p<0.05",
         lambda S: float((S['phase'][S['phase'].struct == 'sign-flip-switch'].wilcoxon_p < 0.05).mean()), 0.97, T_1DP),
      ("W.reroute",  "56% of persistent-reroute cells p<0.05",
         lambda S: float((S['phase'][S['phase'].struct == 'persistent-switch'].wilcoxon_p < 0.05).mean()), 0.56, T_1DP),
      ("W.static",   "static cells p<0.05 fraction approx 0.17",
         lambda S: float((S['phase'][S['phase'].struct == 'static'].wilcoxon_p < 0.05).mean()), 0.17, T_1DP),
      # ---------- seed count ----------
      ("seeds.20", "phase grid uses 20 seeds",
         lambda S: int(S['phase'].n.max()), 20, 0),
      # ---------- deep-domain (V-Dem), 20-seed ensemble ----------
      ("DD.spearman", "mean pairwise Spearman across seeds = 0.85",
         lambda S: float(S['dd']['pairwise_spearman']), 0.85, T_2DP),
      ("DD.var_corr", "E_j vs within-country variance Spearman = 0.03",
         lambda S: float(S['dd']['corr_within_var']), 0.03, T_2DP),
      ("DD.out_corr", "E_j vs out-strength = 0.80",
         lambda S: float(S['dd']['corr_out_strength']), 0.80, T_2DP),
      ("DD.two_corr", "E_j vs two-step reach = 0.78",
         lambda S: float(S['dd']['corr_two_step']), 0.78, T_2DP),
      ("DD.valloss",  "ensemble val MSE = 0.043",
         lambda S: float(S['dd']['val_loss_mean']), 0.043, T_2DP),
      ("DD.suffrage_rank", "suffrage E_j rank = 16 (mean ordering)",
         lambda S: int(S['dd']['ranks']['Suffrage']['ej']), 16, 0),
      ("DD.elected_var_rank", "elected officials highest variance (rank 1)",
         lambda S: int(S['dd']['ranks']['Elected_officials']['var']), 1, 0),
      # ---------- five-domain placement (Table 2): realized departures ----------
      # Democracy realized is the 20-seed ensemble MEAN (paper reports 0.04 +/- 0.03);
      # the single-seed canonical run lands at ~0.038, well within T_1DP of the mean.
      # Tolerance is widened to T_1DP here because this is a seed-ensemble mean of a
      # run-sensitive quantity, NOT a deterministic single fit (unlike the four below).
      ("D.dem.real",  "democracy realized departure = 0.04 (20-seed ensemble mean)",
         lambda S: float(S['dom'].set_index('domain').loc['democracy', 'realized']), 0.04, T_1DP),
      ("D.macro.real", "macro-finance realized departure = 0.15",
         lambda S: float(S['dom'].set_index('domain').loc['macro', 'realized']), 0.15, T_2DP),
      ("D.wdi.real",  "development realized departure approx 0.00",
         lambda S: float(S['dom'].set_index('domain').loc['development', 'realized']), 0.00, T_2DP),
      ("D.rv.real",   "realized-vol realized departure = 0.02",
         lambda S: float(S['dom'].set_index('domain').loc['realized_vol', 'realized']), 0.02, T_2DP),
      ("D.beijing.real", "air-quality realized departure = 0.01",
         lambda S: float(S['dom'].set_index('domain').loc['air_quality', 'realized']), 0.01, T_2DP),
      # ---------- five-domain placement (Table 2): pooled departures ----------
      # NOTE: there is deliberately NO democracy pooled check. The democracy pooled
      # comparison is weakly identified (rank correlation near zero, sign-unstable
      # across retrainings: pooled departure spans ~0.6-1.4). The paper reports no
      # pooled point estimate for democracy, so the verifier asserts none. The four
      # deterministic panels below DO have stable pooled values.
      ("D.macro.pool", "macro-finance pooled departure = 0.12",
         lambda S: float(S['dom'].set_index('domain').loc['macro', 'pooled']), 0.12, T_2DP),
      ("D.wdi.pool",   "development pooled departure = 0.24",
         lambda S: float(S['dom'].set_index('domain').loc['development', 'pooled']), 0.24, T_2DP),
      ("D.rv.pool",    "realized-vol pooled departure = 0.17",
         lambda S: float(S['dom'].set_index('domain').loc['realized_vol', 'pooled']), 0.17, T_2DP),
      ("D.beijing.pool", "air-quality pooled departure = 0.05",
         lambda S: float(S['dom'].set_index('domain').loc['air_quality', 'pooled']), 0.05, T_2DP),
      # ---------- panel dimensions ----------
      ("panel.cy", "democracy training window = 4094 country-years",
         lambda S: int(S['dd']['n_country_years']), 4094, 0),
    ]

def load_sources(root):
    S = {}
    S['phase'] = pd.read_csv(os.path.join(root, 'phase_diagram_results.csv'))
    S['dom']   = pd.read_csv(os.path.join(root, 'domain_placement.csv'))
    with open(os.path.join(root, 'deepdomain_ensemble.json')) as f:
        S['dd'] = json.load(f)
    return S

def run_checks(root, verbose=False):
    S = load_sources(root); checks = build_checks()
    npass = nfail = nerr = 0; lines = []
    for cid, desc, acc, exp, tol in checks:
        try:
            got = acc(S)
            ok = (got == exp) if tol == 0 else (abs(got - exp) <= tol)
            if ok: npass += 1; status = "PASS"
            else:  nfail += 1; status = "FAIL"
            if verbose or not ok:
                lines.append(f"  [{status}] {cid:18s} exp={exp} got={got} (tol {tol}) — {desc}")
        except Exception as e:
            nerr += 1; lines.append(f"  [ERR ] {cid:18s} {type(e).__name__}: {e}")
    return npass, nfail, nerr, lines
