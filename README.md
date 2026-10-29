# Treatment-effect relationships: replication package

This repository reproduces the six simulation figures and the two real-data tables in *Inference for relationships between treatment effects*. It includes the simulation generators, estimators, seeds, analytic data inputs, plotting code, lightweight numerical summaries, and manuscript-format table generation.

## Run

Python 3.12 was used for verification. Install the recorded dependencies in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_replication.py all
```

The default command regenerates **all 5,000 superpopulation datasets**, **all 10,000 finite-population randomizations**, and both complete application analyses. It then rebuilds `tables_figures.pdf` (six simulation figures followed by two application tables), the individual figure files, and LaTeX table bodies. Large generated raw records, prediction arrays and graphical files are created by these commands, rather than duplicated in this compact repository. The simulations use four workers and can take several minutes. Seeds are fixed; parallel scheduling does not change the datasets or assignments. Commands overwrite the corresponding generated results, so retain a separate checkout if comparing with the bundled outputs.

Other entry points:

```bash
python run_replication.py figures       # rebuild displays from saved results
python run_replication.py applications  # refit NVSS and Hillstrom, rebuild displays
python run_replication.py simulations   # rerun both simulation frameworks
python run_replication.py preflight     # small deterministic checks, not a full rerun
```

Run commands from any working directory; paths are relative to the scripts. Full-generation commands do not require historical results. Optional archive comparisons in some scripts are skipped when the historical archive is absent. The repository intentionally omits superseded simulation designs and duplicate historical prediction caches.

## Contents and targets

- `targeted_revision/super/`: superpopulation generator, nuisance estimators, complete runner and frozen seed specification. All five plotted methods are regenerated together: Direct, Calibrated, Naive, X-only, and Oracle EIF. Each of five sample sizes has 1,000 independent datasets. Bias and RMSE use finite point estimates. Coverage uses **all attempted datasets**, with undefined intervals counted as noncoverage. The implementation withholds all three ratio summaries when either direct marginal variance is nonpositive, including the slope and intercept.
- `x_interactions_revision/`: finite-population generator and complete-randomization runner. One fixed population at each of five sizes is randomized 2,000 times. The three plotted methods are OLS, model assistance using `W0=(X,U,X-by-U)`, and Ding. A previously computed additional `W1` comparison remains in the raw results and runner but is not plotted. Here kappa=2 is the finite-population standard deviation (denominator n−1) of each residual individual-effect component after orthogonal projection off X. Ding's variance estimator uses the arm sample covariance of the observed vectors `(X Y1, X Y2)` transformed by the known full-population Gram matrix, divided by each arm size, then the common delta-method gradients.
- `real_x_comparison/`: the 100,000-row NVSS analytic sample, Hillstrom source data, exact analysis code, saved estimates and manuscript table builder. No network download is needed for either analysis.
- `simulation/`, `tables/`: plotting code and numerical input summaries.
- `verification/`: the verification records for this package.

The nominal intervals are pointwise 95% Wald intervals. Displayed variance and correlation intervals are intersected with their parameter spaces; this changes no coverage event for an admissible true value. Empty intersections remain empty. Raw intervals and negative direct variance estimates are preserved in the machine-readable results.

## Data provenance and scope

The NVSS input is restored automatically from losslessly compressed, text-encoded pieces in data_parts/ (verified by SHA-256); it is the same analytic sample used in the manuscript; its SHA-256 is `c3b18686d481048e5d82ad85d18806e8031dfa0e196dd3c705dded035413f4ba`. Earlier calculation records describe simple random sampling without replacement of 100,000 from 252,597 eligible 2019 births, using NumPy `default_rng(20191028)`. The recovered source package did **not** contain the extraction program or the full national fixed-width data file. Thus this package completely reproduces estimation from the bundled analytic sample, while re-extracting that sample from the original national file was not independently verified. The original `source_row` field is retained for provenance.

NVSS treatment is continued versus stopped first-trimester smoking among mothers who smoked before pregnancy. The retained raw out-of-fold propensity predictions range from 0.0076874880 to 1.1398360701; clipping to [0.01,0.99] affects one lower and 2,095 upper predictions. Least-squares boosting can predict above one before clipping. The observed data do not establish unconfoundedness.

The Hillstrom input includes the original three-arm randomized data. The analysis retains 42,613 men's-email and no-email customers. Spending is zero for 42,224 (99.0871%) and its largest value is $499. Treatment-arm filtering, covariate construction, and the outcome scales are implemented in the analysis script.

## Verification and presentation

The original GitHub repository contained only `.gitattributes`; there was no runnable replication package to preserve. The saved source archives were therefore recovered and checked before this package was populated. A missing full superpopulation batch runner was supplied without changing its estimation core. Historical input-cache checks were made optional so the analyses run from the actual bundled data.

The real-data estimates and standard errors reproduce the saved analyses exactly, including all 128 scalar rows across both datasets. Six manuscript cells had previously been rounded twice; the manuscript table generator now scales and rounds once from full-precision results. This is a display correction and does not change any estimate, interval, or scientific result. The simulation figures use the original plotted metrics and retain the original method comparisons.

See `verification/` for full rerun counts, tolerances, original-result comparisons, and figure checks. Numerical reproducibility does not establish the adequacy of boundary-adjacent Wald inference; the applications have weak estimated effect heterogeneity and should be read accordingly.

Verified full rerun: 5,000 superpopulation datasets and 200,000 scalar records, with every coverage indicator identical to the archive (largest point difference 2.85e-14; largest SE difference 7.28e-12). All 10,000 finite assignments and 320,000 scalar estimates were bit-identical; no fits failed. All six regenerated simulation PNGs were pixel-identical. After running the simulations, independently recompute their summaries and exact finite targets:

```bash
python verify_simulations.py
```

The preflight saves separate results and cannot overwrite the full superpopulation output. Generated prediction arrays for the applications are omitted to keep the repository compact; application reruns recreate them before the independent audit.
