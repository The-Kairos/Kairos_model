# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 09:07:20 UTC | qqPwV3d4BZE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.532 | 0.757 | 53.345 | 6.298 | 7.890 | 8.196 | 3.024 |

## 2026-06-28 09:07:20 UTC | qqPwV3d4BZE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/qqPwV3d4BZE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.532` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.757 |
| save_clips | - |
| sample_frames | 0.929 |
| caption_frames | 33.409 |
| sample_fps | 2.173 |
| detect_object_yolo | 8.125 |
| audio_scan | 13.862 |
| asr_timings | 15.205 |
| ast_timings | 24.269 |
| describe_scenes | 6.298 |
| summarize_scenes | 7.890 |
| synthesize_synopsis | 8.196 |
| make_embedding | 3.024 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.344 |
| branch_yolo_total | 10.304 |
| branch_audio_total | 53.345 |
