# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:43:31 UTC | xzmEZo_HUpY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.398 | 0.780 | 51.779 | 11.630 | 7.468 | 9.453 | 3.254 |

## 2026-06-27 04:43:31 UTC | xzmEZo_HUpY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xzmEZo_HUpY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.398` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.091 |
| caption_frames | 38.534 |
| sample_fps | 2.277 |
| detect_object_yolo | 8.726 |
| audio_scan | 11.922 |
| asr_timings | 12.438 |
| ast_timings | 27.411 |
| describe_scenes | 11.630 |
| summarize_scenes | 7.468 |
| synthesize_synopsis | 9.453 |
| make_embedding | 3.254 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.631 |
| branch_yolo_total | 11.009 |
| branch_audio_total | 51.779 |
