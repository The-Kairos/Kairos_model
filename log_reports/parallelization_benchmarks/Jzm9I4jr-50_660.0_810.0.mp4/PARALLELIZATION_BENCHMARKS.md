# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:43:08 UTC | Jzm9I4jr-50_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.040 | 0.680 | 53.454 | 14.981 | 10.171 | 16.466 | 4.038 |

## 2026-06-25 05:43:08 UTC | Jzm9I4jr-50_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Jzm9I4jr-50_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.040` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 1.155 |
| caption_frames | 29.093 |
| sample_fps | 2.144 |
| detect_object_yolo | 8.835 |
| audio_scan | 9.627 |
| asr_timings | 10.920 |
| ast_timings | 32.900 |
| describe_scenes | 14.981 |
| summarize_scenes | 10.171 |
| synthesize_synopsis | 16.466 |
| make_embedding | 4.038 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.253 |
| branch_yolo_total | 10.984 |
| branch_audio_total | 53.454 |
