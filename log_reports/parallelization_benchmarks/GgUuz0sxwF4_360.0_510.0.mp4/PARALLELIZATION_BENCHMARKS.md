# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:01:34 UTC | GgUuz0sxwF4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.072 | 0.804 | 50.688 | 10.618 | 7.948 | 13.114 | 3.269 |

## 2026-06-25 02:01:34 UTC | GgUuz0sxwF4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GgUuz0sxwF4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.072` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 0.974 |
| caption_frames | 36.539 |
| sample_fps | 2.248 |
| detect_object_yolo | 8.494 |
| audio_scan | 13.896 |
| asr_timings | 9.775 |
| ast_timings | 27.009 |
| describe_scenes | 10.618 |
| summarize_scenes | 7.948 |
| synthesize_synopsis | 13.114 |
| make_embedding | 3.269 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.518 |
| branch_yolo_total | 10.748 |
| branch_audio_total | 50.688 |
