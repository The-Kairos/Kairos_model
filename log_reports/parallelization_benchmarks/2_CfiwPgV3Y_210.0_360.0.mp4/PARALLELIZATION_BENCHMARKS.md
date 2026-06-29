# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:42:13 UTC | 2_CfiwPgV3Y_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 218.900 | 0.743 | 55.119 | 26.147 | 52.230 | 31.408 | 3.536 |
| 2026-06-27 15:55:06 UTC | 2_CfiwPgV3Y_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.870 | 0.760 | 56.123 | 9.794 | 6.616 | 12.033 | 3.627 |

## 2026-06-23 14:42:13 UTC | 2_CfiwPgV3Y_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2_CfiwPgV3Y_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `218.900` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.743 |
| save_clips | - |
| sample_frames | 0.949 |
| caption_frames | 36.989 |
| sample_fps | 2.171 |
| detect_object_yolo | 8.241 |
| audio_scan | 14.710 |
| asr_timings | 10.868 |
| ast_timings | 29.532 |
| describe_scenes | 26.147 |
| summarize_scenes | 52.230 |
| synthesize_synopsis | 31.408 |
| make_embedding | 3.536 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.943 |
| branch_yolo_total | 10.419 |
| branch_audio_total | 55.119 |

## 2026-06-27 15:55:06 UTC | 2_CfiwPgV3Y_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2_CfiwPgV3Y_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.870` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 0.958 |
| caption_frames | 37.923 |
| sample_fps | 2.195 |
| detect_object_yolo | 8.432 |
| audio_scan | 15.013 |
| asr_timings | 11.177 |
| ast_timings | 29.925 |
| describe_scenes | 9.794 |
| summarize_scenes | 6.616 |
| synthesize_synopsis | 12.033 |
| make_embedding | 3.627 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.887 |
| branch_yolo_total | 10.633 |
| branch_audio_total | 56.123 |
