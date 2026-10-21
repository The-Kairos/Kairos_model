# MovieSum pilot

Start with **[RESULTS.md](RESULTS.md)**: three- and four-movie averages, all available measures, margins against published zero-shot baselines, and limitations. [results.json](results.json) preserves per-movie P/R/F1 scores, reference checks, scorer settings and hashes.

Completed movies: Baby Driver, Get Out and The Martian; the four-movie comparison adds the earlier Titanic sanity check. Yes Man was skipped. See [input validation](INPUT_VALIDATION.md) for unverified completeness/cut limitations.

Rebuild the concise report without API calls:

```powershell
.venv\Scripts\python.exe evaluation\pilot\report_results.py
```

Raw artifacts stay local and untracked in `local_20261020/`; movie files are in `movies/`, and the official split is `test.jsonl`. Preserve these for provenance and rescoring. The verified snapshot permits report generation without those large files.

The runner preserves checkpoints and skips completed movies:

```powershell
.venv\Scripts\python.exe -u evaluation\pilot\run_pilot.py --run-dir evaluation\pilot\local_20261020 --skip yes_man --allow-unverified-inputs
```

YOLO uses explicit batches of 4, FP32, 640px inference and unchanged 4 FPS sampling. OOM retries restart the scene at batches 2, then 1; atomic scene checkpoints preserve completed work. This is independent of prompt caching. The pilot is finished; restarting is unnecessary for reading or regenerating the report.
