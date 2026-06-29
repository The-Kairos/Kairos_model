# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:43:14 UTC | uErmi6kKVIs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 71.950 | 0.848 | 33.283 | 4.099 | 3.975 | 8.390 | 1.454 |

## 2026-06-27 00:43:14 UTC | uErmi6kKVIs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uErmi6kKVIs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `71.950` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.848 |
| save_clips | - |
| sample_frames | 0.230 |
| caption_frames | 10.382 |
| sample_fps | 1.786 |
| detect_object_yolo | 6.090 |
| audio_scan | 15.743 |
| asr_timings | 9.398 |
| ast_timings | 8.132 |
| describe_scenes | 4.099 |
| summarize_scenes | 3.975 |
| synthesize_synopsis | 8.390 |
| make_embedding | 1.454 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.618 |
| branch_yolo_total | 7.882 |
| branch_audio_total | 33.283 |
