# Titanic sanity check

Evaluate the [revised summary](generated_summary.txt) against the [user-supplied MovieSum reference](reference_summary.txt). See **[RESULTS.md](RESULTS.md)** for measured scores, paper context, and interpretation.

This folder evaluates only the revised summary. The saved narrative, prompt, and metadata document its provenance. `metrics.json` contains revised-only ROUGE, chrF, and length diagnostics.

To reproduce scores from the project root:

```powershell
python evaluation/titanic_sanity/run_summary_check.py
```

Requires `rouge-score`, `nltk`, and existing project dependencies. The script calls the configured model only if `generated_summary.txt` is absent. With the saved output present, scoring requires no API call.

This is a one-movie diagnostic, not a fresh end-to-end benchmark. See the [benchmark plan](../MOVIESUM_BENCHMARK_PLAN.md) for the broader study.
