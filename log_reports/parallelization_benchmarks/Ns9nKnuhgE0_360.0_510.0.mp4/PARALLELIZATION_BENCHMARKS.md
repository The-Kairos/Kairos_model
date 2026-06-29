# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:55:43 UTC | Ns9nKnuhgE0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.221 | 0.814 | 49.142 | 18.999 | 12.251 | 19.748 | 3.713 |

## 2026-06-25 10:55:43 UTC | Ns9nKnuhgE0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ns9nKnuhgE0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.221` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 1.034 |
| caption_frames | 39.720 |
| sample_fps | 2.322 |
| detect_object_yolo | 9.015 |
| audio_scan | 10.827 |
| asr_timings | 8.186 |
| ast_timings | 30.120 |
| describe_scenes | 18.999 |
| summarize_scenes | 12.251 |
| synthesize_synopsis | 19.748 |
| make_embedding | 3.713 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.760 |
| branch_yolo_total | 11.343 |
| branch_audio_total | 49.142 |
