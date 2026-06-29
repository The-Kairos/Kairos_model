# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:24:13 UTC | PfCclequ_ms_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.246 | 0.790 | 51.489 | 20.560 | 33.853 | 19.244 | 3.657 |

## 2026-06-25 14:24:13 UTC | PfCclequ_ms_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PfCclequ_ms_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.246` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 0.937 |
| caption_frames | 40.173 |
| sample_fps | 2.224 |
| detect_object_yolo | 8.897 |
| audio_scan | 12.300 |
| asr_timings | 9.522 |
| ast_timings | 29.658 |
| describe_scenes | 20.560 |
| summarize_scenes | 33.853 |
| synthesize_synopsis | 19.244 |
| make_embedding | 3.657 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.115 |
| branch_yolo_total | 11.127 |
| branch_audio_total | 51.489 |
