# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 20:03:30 UTC | sbirEH3xfpk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.208 | 0.812 | 46.643 | 10.946 | 13.130 | 17.681 | 2.763 |

## 2026-06-26 20:03:30 UTC | sbirEH3xfpk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sbirEH3xfpk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.208` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 0.902 |
| caption_frames | 30.475 |
| sample_fps | 2.148 |
| detect_object_yolo | 7.315 |
| audio_scan | 13.923 |
| asr_timings | 11.678 |
| ast_timings | 21.033 |
| describe_scenes | 10.946 |
| summarize_scenes | 13.130 |
| synthesize_synopsis | 17.681 |
| make_embedding | 2.763 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.383 |
| branch_yolo_total | 9.469 |
| branch_audio_total | 46.643 |
