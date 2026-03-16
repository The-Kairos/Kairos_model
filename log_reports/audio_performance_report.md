# Comprehensive Audio Performance Report

This report evaluates the performance improvements of the parallel audio processing pipeline (ASR and AST) compared to its sequential legacy counterpart.

## Performance Metrics
Using the standard speedup formula $S = \frac{T_s}{T_p}$, where $T_s$ is the sequential execution time (legacy) and $T_p$ is the parallel execution time (new).

**Note**: Timings below are estimated wall times extracted from checkpoints. Times have been rounded and formatted for human readability.

| Video | Video Length | Legacy ASR Time | New ASR Time | ASR Speedup | Legacy AST Time | New AST Time | AST Speedup |
|-------|--------------|-----------------|--------------|-------------|-----------------|--------------|-------------|
| AI beyond language and vision _ Paul Liang _ TEDxMIT.mp4 | 00:16:07.600 | 5 min 19 sec | 35 sec | 9.12x | 1 min 7 sec | 32 sec | 2.08x |
| Argentina v France Full Penalty Shoot-out.mp4 | 00:07:39.000 | 6 min 18 sec | 21 sec | 17.90x | 1 min 43 sec | 50 sec | 2.08x |
| How to Make Pasta - Without a Machine.mp4 | 00:05:28.000 | 4 min 15 sec | 17 sec | 14.98x | 1 min 3 sec | 39 sec | 1.60x |
| NEW YORK TIMES SQUARE 2024 _ 4K WALK TOUR MORNING.mp4 | 00:11:19.917 | 21 min 40 sec | 17 sec | 75.75x | 1 min 9 sec | 39 sec | 1.74x |
| Statistical Learning_ 5.2 K-fold Cross Validation.mp4 | 00:13:33.200 | 4 min 18 sec | 36 sec | 7.04x | 37 sec | 28 sec | 1.31x |
| Titanic.1997.mkv | 03:14:44.631 | 3 hour 48 min | 6 min 58 sec | 32.67x | 32 min 44 sec | 20 min 3 sec | 1.63x |
| Watch Malala Yousafzai's Nobel Peace Prize acceptance speech.mp4 | 00:04:33.473 | 3 min 32 sec | 11 sec | 18.75x | 29 sec | 16 sec | 1.77x |
| Young Sheldon - First Day of High School.mp4 | 00:02:48.502 | 2 min 9 sec | 10 sec | 12.44x | 37 sec | 24 sec | 1.55x |
