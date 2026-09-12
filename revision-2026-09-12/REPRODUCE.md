# Reproduce the September CBDC draft

Tier: research manuscript draft with executable synthetic evidence. Not a field validation, deployed cryptographic system, journal acceptance, or arXiv submission. The intended venue remains Frontiers in Blockchain, RT 80798; this candidate has not been checked for a final portal submission.

Use the repository root. Do not run the old root manuscript or overwrite August results.

## Environment

The computations used Python 3.14.7 and the exact pins in `detection/requirements.txt` (NumPy 2.3.5, SciPy 1.16.3, scikit-learn 1.8.0, pandas 2.3.3). The historical requirements comment names Python 3.14.0; the frozen ladder manifests record 3.14.7, which was also used for the September reproduction. Create a virtual environment and install those pins if necessary. The existing `.venv` is excluded from the review archive.

PDF figures were generated with system Python/matplotlib; `evidence/plot-environment.json` records those versions. Figures are shipped as PDF, so matplotlib is not needed to build the arXiv source archive. A TeX installation with pdfLaTeX, latexmk, BibTeX, and the standard packages listed in main.tex is needed for the PDF.

## Fast numerical regeneration

```bash
.venv/bin/python detection/revision20260912/regenerate.py
python3 detection/revision20260912/plot_figures.py
```

The first command checks all 52 independent-population raw records, seed uniqueness, paired count identities, all 122 ladder records and tier identities, and aggregates 90% pointwise Student-t intervals. It writes `revision-2026-09-12/evidence/adjudicated-results.json` and the manuscript table inputs. It preserves every August raw result. Its inference uses a separate direct t-formula as a cross-check on the old summary.

`regenerate.py` is not a full data-generation replay. The original records contain outcomes/configurations, not all generated transaction populations. The generator and original seeds produce those populations on demand.

## Full independent-population reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python detection/confirmatory/run_confirmatory.py --workers 6 --out-dir revision-2026-09-12/evidence/confirmatory-rerun
.venv/bin/python detection/revision20260912/regenerate.py
```

This repeats the original 52 train/test seed pairs, not 52 new independent replications. Use a fresh output directory if retaining the shipped reproduction. Raw records match the original exactly. H2 is now `DESCRIPTIVE_ONLY`; an unset tolerance cannot produce `NON_INFERIOR`. A separate new-seed study would be additional research, not reproduction.

## Retrospective controls and original ladder replay

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python detection/revision20260912/run_diagnostics.py
```

Read `detection/revision20260912/SUPPLEMENTAL_PLAN.md` first. The script deliberately refuses to overwrite an existing `permutation-controls.jsonl`. For a fresh full replay, preserve or move the shipped file first. For interrupted work, `--resume` validates every completed seed/configuration, skips both recorded successes and failures, and executes only unrecorded units. Run only one writer at a time. It generates 22 new worlds, permutes labels consistently before folds/tuning, evaluates all four models, and finally exactly replays original ladder seed 2026082051. The run began with four workers, preserved completed checkpoints after a compute interruption, and resumed remaining units with 12 workers. No completed unit was rerun, no seed changed, and no record was overwritten. `ladder-replay.json` compares every scientific output field and chosen hyperparameter against the August record.

The older `run_ladder.py` driver resumes/skips completed units and overwrites its manifest. For full original-ladder replay, use a separate checkout/archive copy and move its `detection/results/ladder/frozen` directory aside before running all four frozen packages. Do not infer a replay occurred from a successful command that skipped every existing unit.

## Focused checks and simulator

```bash
.venv/bin/python detection/confirmatory/test_inference.py
.venv/bin/python detection/confirmatory/test_endpoint.py
.venv/bin/python detection/confirmatory/test_lock.py
.venv/bin/python detection/test_pipeline.py
.venv/bin/python pet_aml_sim.py --days 2 --seed 7
.venv/bin/python pet_aml_sim.py --days 2 --seed 7 --tier-amounts
```

There are 4 gatekeeping regression tests, 11 endpoint tests, 11 lock tests, and 2 generator-audit tests. Simulator summaries are compared by whole configuration, not selected rows. Timings are modeled queue/service quantities, not measured cryptographic throughput.

## Manuscript and arXiv archive

```bash
cd revision-2026-09-12/manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The arXiv archive contains only the top-level manuscript, bibliography, generated table fragments, PDF figures, and a compiled bibliography (`main.bbl`). It was extracted into an empty durable scratch directory and built there. No path outside that extracted directory is needed. The code review archive is separate from the minimal arXiv manuscript archive.
