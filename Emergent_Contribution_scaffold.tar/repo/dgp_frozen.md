# Frozen data-generating process — phase-diagram experiment

This document records the synthetic data-generating process (DGP) behind the
phase diagram (Table 1 and Figure 1), together with the predictions fixed before
the run. It is a faithful transcription of the phase-diagram cell in
`notebooks/Paper1_reproducibility.ipynb`; the cell, not this document, is the
authoritative source, and the two are kept in sync. The DGP and its predictions
were fixed before the grid was run; the constants below are marked in the code
as frozen calibration and were not adjusted after seeing results.

## The system

Each cell of the grid simulates a two-regime, discrete-time nonlinear system on
`N = 7` components:

    x_{t+1} = A_r x_t + alpha * tanh(B_r x_t) + epsilon_t

where `r in {0, 1}` indexes the active regime at time `t`, `alpha` is the
nonlinearity strength, and `epsilon_t` is i.i.d. Gaussian noise. A single node
(`DRIVER = 0`) carries the regime-dependent structure.

### Frozen constants (do not change)

| Constant   | Value | Role                                                          |
|------------|-------|---------------------------------------------------------------|
| `N`        | 7     | number of components                                          |
| `HORIZON`  | 8     | finite horizon T for E_j, AC, and ground-truth contribution   |
| `DRIVER`   | 0     | the node whose downstream pathway changes across regimes      |
| `BGAIN`    | 2.0   | driver pathway gain in the active regime                      |
| `DIAG`     | 0.6   | mean diagonal (self-persistence) of the linear part A         |
| `NOISE`    | 0.5   | s.d. of the additive Gaussian innovation                      |
| spec. cap  | 0.85  | each A is rescaled so its spectral radius does not exceed 0.85|
| `T`        | 400   | simulated trajectory length per cell                          |

### Regime construction (`make_regimes`)

Both regimes share a base linear map `A` (off-diagonal entries ~ N(0, 0.04^2),
diagonal ~ Uniform(`DIAG`-0.05, `DIAG`+0.05)) and a base `B` (~ N(0, 0.04^2)).
Regime 1 sets the driver's downstream lag rows (`B1[1..3, DRIVER] = BGAIN`). The
second regime depends on the structure axis:

- **static** — only regime 0 is ever used.
- **smooth-drift** — regime 2 keeps the driver pathway at `+BGAIN`; the system
  interpolates linearly from regime 0 to regime 1 over the trajectory. This is
  the negative control: time-variation *without* regime change.
- **persistent-switch** ("reroute") — regime 2 zeroes the driver pathway and
  moves the gain to a different node (`B2[1..3, 4] = BGAIN`): a driver change
  without sign reversal.
- **sign-flip-switch** ("sign reversal") — regime 2 flips the driver pathway to
  `-BGAIN`, so the time-average cancels the driver's contribution.

Switching (for the two non-smooth structures) is a Bernoulli process with
per-step probability set by the persistence axis.

## The grid (144 cells = 4 x 3 x 3 x 4)

| Axis            | Values (key: value)                                         |
|-----------------|-------------------------------------------------------------|
| regime structure| static, smooth-drift, persistent-switch, sign-flip-switch   |
| nonlinearity    | low: 0.3, med: 0.8, high: 1.5  (the `alpha` above)          |
| persistence     | short: p=0.30, medium: p=0.12, long: p=0.02 (switch prob.)  |
| amplitude       | small: 0.02, med: 1.0, large: 6.0, extreme: 12.0 (probe eps)|

Each cell is run over **20 seeds** (`SEEDS = range(20)`). Simulation seeds the
trajectory RNG at `seed + 7`; regime construction seeds at `seed`.

## Metrics computed per cell

For each cell we compute, over the 20 seeds:

- `ej_t`  = Spearman(E_j, ground-truth contribution)
- `acr_t` = Spearman(AC-realized, ground truth)   — AC of the mean realized Jacobian
- `acp_t` = Spearman(AC-pooled, ground truth)      — AC of a single mean-state linearization
- `ej_acr`= Spearman(E_j, AC-realized)
- `d_real`= ej_t - acr_t, with a 2000-sample paired bootstrap 95% CI
- `d_pool`= ej_t - acp_t (the headline fidelity advantage), with the same CI
- `wilcoxon_p` = paired Wilcoxon p-value (E_j vs AC-pooled) when n >= 6

Ground-truth contribution (`true_contrib`) is finite-amplitude
perturb-and-propagate through the **true** nonlinear dynamics (not the
estimator): for each node, perturb by the cell's amplitude `eps`, propagate the
perturbed and baseline trajectories through the actual regime-specific maps over
the horizon, and accumulate squared divergence. This is the reference the
measures are scored against; fidelity to it — not the bare E_j-vs-AC gap — is the
primary quantity.

## Predictions fixed in advance

The four-region structure was predicted before the grid was run:

1. **Controllability-sufficient** (static, smooth-drift): E_j, both AC baselines,
   and ground truth agree (`d_pool` approx 0) at all nonlinearity levels.
   Smooth-drift is the negative control — time-variation alone should not
   separate the measures.
2. **Intermediate** (persistent-switch / reroute): a real but smaller advantage
   for E_j (`d_pool` positive but modest).
3. **E_j separates** (sign-flip-switch / sign reversal): E_j holds high fidelity
   while pooled AC collapses, the largest `d_pool`, strongest at higher
   nonlinearity.
4. **Both fail** (extreme amplitude): E_j's own local-linearization fidelity
   erodes and the measures converge, bounding where the measure is the right tool.

The committed `results/phase_diagram_results.csv` realizes these predictions; the
paper's Table 1 reads its `d_pool` column and the prose reads the Wilcoxon
fractions and CIs.

## Reproducing

Running the phase-diagram cell of the notebook regenerates
`phase_diagram_results.csv` (144 rows) and `phase_diagram.png` from the constants
above. The DGP uses only NumPy's default RNG seeded as described, so the grid is
deterministic given the seed set and reproduces across machines.
