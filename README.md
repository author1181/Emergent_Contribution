# Trajectory-Aware Node Contributions and the Limits of Static Controllability

Reproducibility repository for the ICDM 2026 submission **"Trajectory-Aware
Node Contributions and the Limits of Static Controllability"** (paper #<submission-id>).

This work defines **emergent contribution** ($E_j$), a finite-horizon measure of
a node's dynamical leverage in an estimated nonlinear, time-varying system,
computed from the ordered product of a fitted model's Jacobians along its
trajectory. In the linear time-invariant limit it reduces *exactly* to average
controllability, generalizing that measure rather than competing with it. The
repository reproduces the paper's three empirical pillars: the LTI-limit
reduction, a pre-registered synthetic **phase diagram** characterizing when the
measure departs from average controllability, and a five-domain real-data
**placement** with a deep-domain worked example on a Varieties of Democracy
(V-Dem) panel.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/author1181/Emergent_Contribution/blob/main/notebooks/Paper1_reproducibility.ipynb)

The paper is included under [`paper/`](paper): `paper1.pdf`.

## Repository structure

```
Emergent_Contribution/
├── README.md                         This file
├── LICENSE                           CC-BY-4.0
├── requirements.txt                  Python dependencies
├── dgp_frozen.md                     Frozen synthetic DGP + pre-registered predictions
├── paper/
│   ├── paper1.pdf                    Main paper
├── notebooks/
│   └── Paper1_reproducibility.ipynb  End-to-end Colab orchestrator (run this)
├── src/                              Pipeline modules (imported by the notebook)
│   ├── NAVAR.py                      Neural additive VAR model (MLP/Conv + LSTM)
│   ├── dataloader.py                 NAVAR data loader 
│   ├── train_NAVAR.py               NAVAR training loop -> fitted model
│   ├── asof_jacobian.py             Companion Jacobians + stability diagnostics
│   ├── emergent_contribution.py     E_j and average-controllability Gramian trace
│   ├── leakage_safe_split.py        Frozen pre-cutoff temporal split + standardizer
│   ├── beijing_prep.py              Air-quality panel preprocessing
│   ├── beijing_estimator.py         Air-quality panel estimator helper
│   ├── verifier_core.py             CHECKS list mapping paper numbers -> artifacts
│   └── verify_paper_numbers.py      Runner: asserts every paper number (Path A)
├── data/
│   ├── my_data_75_years.csv         V-Dem panel: 89 countries x 75 years, 16 components
│   ├── gmd_macro_core.csv           Global Macro Database core series (macro-finance)
│   ├── rv_dataset.csv               Realized volatilities, 8 global equity indices
│   ├── wdi_reversal_panel.csv       World Development Indicators panel
│   ├── PRSA_Data_20130301-20170228/  Beijing air-quality site CSVs (PRSA_Data_*.csv)
│   └── README.md                    Data dictionary + provenance/citations
└── results/                         Canonical outputs reproduced by the notebook
    ├── phase_diagram_results.csv    20-seed phase grid (Table 1, Fig. 1)
    ├── phase_diagram.png            Phase diagram figure (Fig. 1)
    ├── domain_placement.csv         Five-domain placement (Table 2)
    └── deepdomain_ensemble.json     20-seed deep-domain ensemble (worked example)
```

## How to reproduce

The pipeline is designed to run end-to-end on **Google Colab** with a GPU.

1. Open `notebooks/Paper1_reproducibility.ipynb` in Colab (click the badge
   above, or File → Open notebook → GitHub → this repository).
2. Enable a GPU: **Runtime → Change runtime type → GPU**.
3. **Run the first setup cell.** It clones this repository into the Colab
   session, puts `src/` on the Python path, and points the data and output
   locations at the clone. No manual uploads and no Google Drive are required.
4. **Run the remaining cells top to bottom.** The notebook trains the V-Dem
   NAVAR, runs the five-domain placement, regenerates the phase diagram, and
   runs the deep-domain seed ensemble, writing all artifacts to `results/`.

### Headline results and the command to reproduce them

Running Path A (below) checks all of the following against the committed
artifacts in `results/`:

| Result (paper) | Value | Source artifact |
|---|---|---|
| Phase diagram, sign-reversal Δ (med / high nonlin.) | 0.47 / 0.58 | `phase_diagram_results.csv` |
| Phase diagram, persistent-reroute Δ (high nonlin.) | 0.10 | `phase_diagram_results.csv` |
| Phase diagram, static / smooth-drift Δ | ≈ 0.05 / ≈ 0 | `phase_diagram_results.csv` |
| Sign-reversal cells significant (Wilcoxon p<0.05) | 97% | `phase_diagram_results.csv` |
| Table 2, macro-finance realized departure | 0.09 ± 0.04 | `domain_placement.csv` |
| Table 2, democracy realized departure | 0.04 ± 0.03 | `domain_placement.csv` |
| Table 2, three physical panels (dev / vol / air) | ≤ 0.03 (near-zero) | `domain_placement.csv` |
| Deep-domain mean pairwise Spearman / val MSE | 0.85 / 0.043 | `deepdomain_ensemble.json` |
| Deep-domain E_j vs within-variance correlation | 0.03 | `deepdomain_ensemble.json` |

Exact command (terminal):

```
python src/verify_paper_numbers.py --root results/ --verbose
```

In a Colab cell (uses the `CODE`/`RESULTS` paths set by the setup cell):

```python
!cd {CODE} && python verify_paper_numbers.py --root {RESULTS}/ --verbose
```

### Runtime and compute infrastructure

Reported runtimes are for a single Google Colab GPU session (the development
runs used T4 and A100 GPUs; any CUDA GPU is sufficient, and the verifier runs on
CPU). Approximate wall-clock times for a full Path B re-run:

| Stage | Approx. runtime |
|---|---|
| Setup + V-Dem NAVAR training (single model) | ~1–2 min |
| Five-domain placement (V-Dem, macro, WDI, realized-vol) | ~2–3 min |
| Five-domain placement, air-quality (Beijing) panel | **~30 min** (hourly multi-site data; it is working, not stuck) |
| Phase diagram (20-seed synthetic grid) | ~3–5 min |
| Deep-domain seed ensemble (20 seeds) | ~5–8 min |
| Path A verification | <1 min (CPU) |

The air-quality panel dominates the total; a full re-run is roughly 45 minutes,
most of it Beijing. Path A verification alone requires no GPU and finishes in
seconds.

### Two ways to check the results

- **Path A — verify (≈30 s, no GPU).** After cloning, run the verifier against
  the canonical artifacts in `results/`. From a terminal:

  ```
  python src/verify_paper_numbers.py --root results/
  ```

  **In a Colab notebook cell**, prefix with `!` and run from the `src/`
  directory so the local import resolves (the setup cell defines `CODE` and
  `RESULTS`):

  ```python
  !cd {CODE} && python verify_paper_numbers.py --root {RESULTS}/ --verbose
  ```

  It prints a PASS/FAIL line per check and a summary. `--verbose` shows every
  check; without it, only failures are printed.

- **Path B — re-run (GPU).** Execute the notebook end to end to regenerate the
  artifacts from scratch, then run Path A against your freshly produced
  `results/`. Note that the air-quality (Beijing) panel re-trains on hourly
  multi-site data and takes roughly 30 minutes on a Colab GPU — it is working,
  not stuck.

### What the verifier covers, and how

The verifier maps the paper's reported numbers to the committed artifacts using
two kinds of check:

- **Exact / tolerance checks** for the quantities that are stable and
  reproducible: the synthetic phase diagram (Table 1 cells, confidence
  intervals, seed-robustness fractions) and the deep-domain ensemble
  (correlations, validation loss, rank anchors, panel dimensions).
- **Range-membership checks** for the five real-data departures (Table 2). These
  gaps are small, seed- and hardware-sensitive quantities — neural fits are not
  bit-reproducible across GPUs — so the paper reports each as a seed-ensemble
  mean ± s.d., and the verifier checks that a value falls within the reported
  ensemble range rather than matching a point. A run on different hardware
  therefore reproduces as long as its values land in-range; an out-of-range
  value (a genuine error) still fails. Two additional checks assert the
  substantive gradient that holds across all seeds: macro-finance is the largest
  realized departure, and the three physical panels sit in the near-zero region.

The verifier does not re-run the pipeline (Path B is the defense against
pipeline regressions), does not check the pooled baseline (which is too
seed-variable to report as a point; see below), and does not check single-
instance illustrative figures quoted inline in the prose.

## Reproducibility notes

- **Determinism and seed sensitivity.** The notebook's setup section fixes all
  RNG seeds (Python, NumPy, PyTorch, CUDA/cuDNN) via `seed_everything`, so a run
  is reproducible *within* a given environment. The synthetic phase diagram and
  the deep-domain ordering are stable across environments. The five real-data
  departures, however, are small quantities whose underlying neural fits are not
  bit-reproducible across different GPUs; their values vary by a few hundredths
  across seeds and hardware. The paper therefore reports them as seed-ensemble
  means ± s.d. (8 seeds; 3 for air quality, 20 for democracy), and the verifier
  checks range membership. If your re-run gives, say, macro-finance = 0.12 where
  the paper centers on 0.09, that is expected and in-range — not a failure.
- **The robust finding.** What is stable across seeds and hardware is the
  *gradient*: the two regime-changing panels (macro-finance ≈ 0.09, democracy
  ≈ 0.04) depart from average controllability more than the three near-zero
  physical panels (development, realized volatility, air quality, all ≤ 0.03 in
  every seed). The paper's argument rests on this separation, not on any single
  value.
- **Two controllability baselines.** Departure is measured against a *realized*
  baseline (average of the trajectory Jacobians) and a *pooled* baseline (a
  single mean-state linearization). The paper leads on the realized baseline,
  the conservative apples-to-apples comparison. The pooled comparison is even
  more seed-variable on every panel (e.g. development pooled spans ≈ 0.02–0.33
  across seeds; democracy pooled rank correlation spans −0.4 to +0.4), so no
  pooled point estimates are reported or tabulated, and the verifier does not
  check them.
- **Frozen synthetic DGP.** The phase-diagram data-generating process is fixed
  (`BGAIN=2.0`, `DIAG=0.6`, `NOISE=0.5`) and reported at 20 seeds. The full
  specification and the predictions fixed before running are in
  [`dgp_frozen.md`](dgp_frozen.md).

## Code attribution

The NAVAR architecture and training code — `src/NAVAR.py`, `src/train_NAVAR.py`, `src/dataloader.py` —
is adapted from:
> Bussmann, B., Nys, J., and Latré, S. 2021. Neural Additive Vector
> Autoregression Models for Causal Discovery in Time Series. In *Discovery
> Science*, LNCS vol. 12986. Springer. DOI: 10.1007/978-3-030-88942-5_27.

Original code: <https://github.com/bartbussmann/NAVAR>. Modifications: GPU device
handling; a 4-value return exposing the trained model; panel-aware loss with
country-segment boundaries; and a validation-loss recomputation pass.

The remaining modules — `asof_jacobian.py`, `emergent_contribution.py`,
`leakage_safe_split.py`, `beijing_prep.py`, `beijing_estimator.py`, and the
verifier — are new and developed for this work.

## License

Except where noted, this repository's contents are released under the
[Creative Commons Attribution 4.0 International License](LICENSE) (CC BY 4.0).
The NAVAR-derived modules remain subject to their original license (see Code
attribution). `data/my_data_75_years.csv` is derived from the Varieties of
Democracy (V-Dem) v15 dataset (V-Dem Institute, CC BY 4.0); see `data/README.md`
for full provenance and per-panel source citations.
