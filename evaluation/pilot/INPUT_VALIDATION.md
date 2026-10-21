# Pilot input validation

Actual video/audio endpoints are shorter than subtitle-inflated container durations:

| Movie | Container duration | Actual audiovisual end |
|---|---:|---:|
| Yes Man | 1:53:08 | 0:58:17 |
| Baby Driver | 1:52:45 | 1:37:30 |
| Get Out | 1:44:10 | 1:26:43 |
| The Martian | 2:27:48 | 2:17:55 |

The user excluded Yes Man and authorized the other three as diagnostic inputs. Their cuts/completeness remain unverified. All three subsequently completed; see [RESULTS.md](RESULTS.md).

The full packet scan is preserved locally in `local_20261020/input_checks/packet_ranges.json`. Decoder warnings are retained in the pipeline logs. These results must not be described as a verified full-movie held-out benchmark. Use a new run directory for replacement movie files so old checkpoints are not reused.
