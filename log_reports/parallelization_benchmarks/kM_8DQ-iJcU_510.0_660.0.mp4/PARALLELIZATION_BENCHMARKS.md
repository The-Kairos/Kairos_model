# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:45:43 UTC | kM_8DQ-iJcU_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.066 | 0.675 | 62.511 | 28.007 | 26.155 | 20.710 | 2.856 |

## 2026-06-26 13:45:43 UTC | kM_8DQ-iJcU_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kM_8DQ-iJcU_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.066` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.675 |
| save_clips | - |
| sample_frames | 0.765 |
| caption_frames | 26.575 |
| sample_fps | 1.998 |
| detect_object_yolo | 7.315 |
| audio_scan | 5.489 |
| asr_timings | 35.395 |
| ast_timings | 21.618 |
| describe_scenes | 28.007 |
| summarize_scenes | 26.155 |
| synthesize_synopsis | 20.710 |
| make_embedding | 2.856 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.346 |
| branch_yolo_total | 9.319 |
| branch_audio_total | 62.511 |
