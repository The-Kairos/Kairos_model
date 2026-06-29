# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:21:37 UTC | 0H2xFsZtjNY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 100.573 | 0.823 | 43.850 | 7.236 | 4.098 | 8.738 | 2.294 |

## 2026-06-27 13:21:37 UTC | 0H2xFsZtjNY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0H2xFsZtjNY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `100.573` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.823 |
| save_clips | - |
| sample_frames | 0.837 |
| caption_frames | 22.398 |
| sample_fps | 2.097 |
| detect_object_yolo | 6.804 |
| audio_scan | 11.734 |
| asr_timings | 16.270 |
| ast_timings | 15.837 |
| describe_scenes | 7.236 |
| summarize_scenes | 4.098 |
| synthesize_synopsis | 8.738 |
| make_embedding | 2.294 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.241 |
| branch_yolo_total | 8.907 |
| branch_audio_total | 43.850 |
