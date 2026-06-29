# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:21:25 UTC | Z3F41bP1Lcs_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.190 | 0.564 | 39.621 | 13.053 | 21.333 | 12.208 | 3.507 |

## 2026-06-25 22:21:25 UTC | Z3F41bP1Lcs_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Z3F41bP1Lcs_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.190` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.564 |
| save_clips | - |
| sample_frames | 0.926 |
| caption_frames | 38.688 |
| sample_fps | 1.897 |
| detect_object_yolo | 8.004 |
| audio_scan | 3.351 |
| asr_timings | 0.000 |
| ast_timings | 30.217 |
| describe_scenes | 13.053 |
| summarize_scenes | 21.333 |
| synthesize_synopsis | 12.208 |
| make_embedding | 3.507 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.621 |
| branch_yolo_total | 9.907 |
| branch_audio_total | 33.576 |
