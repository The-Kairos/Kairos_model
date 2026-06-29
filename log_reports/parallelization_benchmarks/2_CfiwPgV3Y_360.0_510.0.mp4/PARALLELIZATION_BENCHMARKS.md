# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:45:38 UTC | 2_CfiwPgV3Y_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 203.868 | 0.786 | 58.782 | 32.696 | 14.489 | 30.983 | 4.305 |
| 2026-06-24 08:46:40 UTC | 2_CfiwPgV3Y_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.067 | 0.785 | 60.928 | 25.353 | 22.338 | 23.918 | 4.276 |

## 2026-06-23 14:45:38 UTC | 2_CfiwPgV3Y_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2_CfiwPgV3Y_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `203.868` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.284 |
| caption_frames | 47.280 |
| sample_fps | 2.406 |
| detect_object_yolo | 9.486 |
| audio_scan | 14.730 |
| asr_timings | 8.887 |
| ast_timings | 35.157 |
| describe_scenes | 32.696 |
| summarize_scenes | 14.489 |
| synthesize_synopsis | 30.983 |
| make_embedding | 4.305 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.570 |
| branch_yolo_total | 11.898 |
| branch_audio_total | 58.782 |

## 2026-06-24 08:46:40 UTC | 2_CfiwPgV3Y_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2_CfiwPgV3Y_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.067` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.263 |
| caption_frames | 31.274 |
| sample_fps | 2.410 |
| detect_object_yolo | 9.415 |
| audio_scan | 16.194 |
| asr_timings | 9.252 |
| ast_timings | 35.474 |
| describe_scenes | 25.353 |
| summarize_scenes | 22.338 |
| synthesize_synopsis | 23.918 |
| make_embedding | 4.276 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.542 |
| branch_yolo_total | 11.831 |
| branch_audio_total | 60.928 |
