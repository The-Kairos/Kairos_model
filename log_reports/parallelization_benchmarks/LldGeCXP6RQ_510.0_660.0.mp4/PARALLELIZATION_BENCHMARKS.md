# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:31:49 UTC | LldGeCXP6RQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 214.047 | 0.774 | 68.345 | 25.532 | 18.991 | 23.778 | 5.129 |

## 2026-06-25 07:31:49 UTC | LldGeCXP6RQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LldGeCXP6RQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `214.047` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 1.558 |
| caption_frames | 55.310 |
| sample_fps | 2.438 |
| detect_object_yolo | 10.783 |
| audio_scan | 14.877 |
| asr_timings | 13.996 |
| ast_timings | 39.463 |
| describe_scenes | 25.532 |
| summarize_scenes | 18.991 |
| synthesize_synopsis | 23.778 |
| make_embedding | 5.129 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.874 |
| branch_yolo_total | 13.226 |
| branch_audio_total | 68.345 |
