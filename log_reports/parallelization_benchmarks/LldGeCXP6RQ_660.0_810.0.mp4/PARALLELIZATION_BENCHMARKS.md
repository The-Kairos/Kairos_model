# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:37:09 UTC | LldGeCXP6RQ_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.351 | 0.760 | 50.586 | 18.563 | 35.632 | 27.541 | 3.113 |

## 2026-06-25 07:37:09 UTC | LldGeCXP6RQ_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LldGeCXP6RQ_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.351` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 0.976 |
| caption_frames | 34.373 |
| sample_fps | 2.157 |
| detect_object_yolo | 8.230 |
| audio_scan | 14.924 |
| asr_timings | 11.769 |
| ast_timings | 23.885 |
| describe_scenes | 18.563 |
| summarize_scenes | 35.632 |
| synthesize_synopsis | 27.541 |
| make_embedding | 3.113 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.355 |
| branch_yolo_total | 10.393 |
| branch_audio_total | 50.586 |
