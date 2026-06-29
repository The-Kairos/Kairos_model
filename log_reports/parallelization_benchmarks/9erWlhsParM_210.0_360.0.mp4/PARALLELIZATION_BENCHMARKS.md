# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:12:33 UTC | 9erWlhsParM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.140 | 0.629 | 44.105 | 17.420 | 7.397 | 14.132 | 2.790 |

## 2026-06-24 18:12:33 UTC | 9erWlhsParM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9erWlhsParM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.140` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.629 |
| save_clips | - |
| sample_frames | 0.725 |
| caption_frames | 29.596 |
| sample_fps | 1.883 |
| detect_object_yolo | 8.070 |
| audio_scan | 11.741 |
| asr_timings | 10.820 |
| ast_timings | 21.535 |
| describe_scenes | 17.420 |
| summarize_scenes | 7.397 |
| synthesize_synopsis | 14.132 |
| make_embedding | 2.790 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.327 |
| branch_yolo_total | 9.959 |
| branch_audio_total | 44.105 |
