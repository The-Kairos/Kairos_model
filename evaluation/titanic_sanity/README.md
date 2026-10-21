# Titanic sanity check

Evaluate the [revised summary](generated_summary.txt) against the [MovieSum reference](reference_summary.txt), verified on 2026-10-21 to match the official test-split Wikipedia plot summary after whitespace normalization. See **[RESULTS.md](RESULTS.md)** for measured scores and the [four-movie zero-shot comparison](../pilot/RESULTS.md) for Titanic alongside the local pilots.

This folder evaluates only the revised summary. The saved narrative, prompt, and metadata document its provenance. `metrics.json` contains revised-only ROUGE, chrF, and length diagnostics.

To reproduce scores from the project root:

```powershell
python evaluation/titanic_sanity/run_summary_check.py
```

Requires `rouge-score`, `nltk`, and existing project dependencies. With the saved output present, scoring requires no API call. To archive the previous result and regenerate with a validated 650-750-word limit, run:

```powershell
python evaluation/titanic_sanity/run_summary_check.py --regenerate
```

Generation uses only the saved narrative and allows up to five attempts to meet the length range; it never receives the reference. A failed generation leaves the saved summary intact.

This is a one-movie diagnostic, not a fresh end-to-end benchmark. See the [benchmark plan](../MOVIESUM_BENCHMARK_PLAN.md) for the broader study.
