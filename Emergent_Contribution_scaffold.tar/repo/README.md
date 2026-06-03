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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/author0725/Emergent_Contribution/blob/main/notebooks/Paper1_reproducibility.ipynb)

The paper and supplementary materials are included under [`paper/`](paper):
the main paper (`paper1.pdf`) and supplement (`paper1_supplementary.pdf`).

## Repository structure

```
Emergent_Contribution/
├── README.md                         This file
├── LICENSE                           CC-BY-4.0
├── requirements.txt                  Python dependencies
├── dgp_frozen.md                     Frozen synthetic DGP + pre-registered predictions
├── paper/
│   ├── paper1.pdf                    Main paper
│   └── paper1_supplementary.pdf      Supplementary materials
├── notebooks/
│   └── Paper1_reproducibility.ipynb  End-to-end Colab orchestrator (run this)
├── src/                              Pipeline modules (imported by the notebook)
│   ├── NAVAR.py                      Neural additive VAR model (MLP/Conv + LSTM)
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

### Two ways to check the results

- **Path A — verify (≈30 s, no GPU).** After cloning, run
  `python src/verify_paper_numbers.py --root results/`. This asserts every
  number reported in the paper against the committed canonical artifacts and
  prints a PASS/FAIL line per check.
- **Path B — re-run (GPU).** Execute the notebook end to end to regenerate the
  artifacts from scratch, then run Path A against your freshly produced
  `results/`.

### What the verifier covers

The verifier checks that the paper's **reported numbers match the committed
artifacts** (Table 1 phase cells, confidence intervals, and seed-robustness
fractions; Table 2 realized and pooled departures; and the deep-domain
ensemble's correlations, validation loss, and rank anchors). It does not re-run
the pipeline; Path B is the defense against pipeline regressions. Single-instance
illustrative figures quoted inline in the prose are not part of the verifier.

## Reproducibility notes

- **Determinism.** The notebook's setup section fixes all RNG seeds (Python,
  NumPy, PyTorch, CUDA/cuDNN) via `seed_everything`. Single-model training uses
  a fixed canonical seed; the deep-domain ensemble varies the seed deliberately
  and seeds each member. The canonical single-model seed is fixed for
  reproducibility; the democracy realized departure is reported as the 20-seed
  ensemble mean, and the single seeded model is a representative instance.
- **Two controllability baselines.** Table 2 reports departure of the node
  ordering from average controllability under a *realized* baseline (average of
  the trajectory Jacobians) and a *pooled* baseline (a single linearization at
  the mean state). The realized baseline is the conservative apples-to-apples
  comparison the paper leads on. For the democracy panel the pooled comparison
  is weakly identified (its rank correlation is near zero and sign-unstable
  across retrainings), so no pooled point estimate is reported for that panel.
- **Frozen synthetic DGP.** The phase-diagram data-generating process is fixed
  (`BGAIN=2.0`, `DIAG=0.6`, `NOISE=0.5`) and reported at 20 seeds. The full
  specification and the predictions fixed before running are in
  [`dgp_frozen.md`](dgp_frozen.md).

## Code attribution

The NAVAR architecture and training code — `src/NAVAR.py`, `src/train_NAVAR.py` —
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
