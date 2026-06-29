# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:11:38 UTC | dcAFJS34zkE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 200.092 | 0.782 | 90.431 | 11.710 | 12.639 | 13.919 | 4.732 |

## 2026-06-26 03:11:38 UTC | dcAFJS34zkE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dcAFJS34zkE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `200.092` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.356 |
| caption_frames | 51.065 |
| sample_fps | 2.386 |
| detect_object_yolo | 9.665 |
| audio_scan | 11.885 |
| asr_timings | 40.937 |
| ast_timings | 37.601 |
| describe_scenes | 11.710 |
| summarize_scenes | 12.639 |
| synthesize_synopsis | 13.919 |
| make_embedding | 4.732 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.427 |
| branch_yolo_total | 12.057 |
| branch_audio_total | 90.431 |
