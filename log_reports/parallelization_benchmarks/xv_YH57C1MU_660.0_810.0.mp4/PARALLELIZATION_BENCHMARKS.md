# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:35:04 UTC | xv_YH57C1MU_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 111.934 | 0.654 | 46.250 | 8.000 | 5.958 | 7.033 | 2.829 |

## 2026-06-27 04:35:04 UTC | xv_YH57C1MU_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xv_YH57C1MU_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `111.934` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.654 |
| save_clips | - |
| sample_frames | 0.952 |
| caption_frames | 29.311 |
| sample_fps | 1.996 |
| detect_object_yolo | 7.532 |
| audio_scan | 14.026 |
| asr_timings | 10.490 |
| ast_timings | 21.725 |
| describe_scenes | 8.000 |
| summarize_scenes | 5.958 |
| synthesize_synopsis | 7.033 |
| make_embedding | 2.829 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.269 |
| branch_yolo_total | 9.533 |
| branch_audio_total | 46.250 |
