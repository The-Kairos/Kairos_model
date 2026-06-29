# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:16:19 UTC | 1IDUWll4TXo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.225 | 0.763 | 51.453 | 29.918 | 16.251 | 29.677 | 4.058 |
| 2026-06-27 14:54:24 UTC | 1IDUWll4TXo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.718 | 0.780 | 51.586 | 8.476 | 11.914 | 9.028 | 3.900 |

## 2026-06-23 13:16:19 UTC | 1IDUWll4TXo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1IDUWll4TXo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.225` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.763 |
| save_clips | - |
| sample_frames | 1.546 |
| caption_frames | 43.264 |
| sample_fps | 2.381 |
| detect_object_yolo | 8.540 |
| audio_scan | 11.629 |
| asr_timings | 7.406 |
| ast_timings | 32.410 |
| describe_scenes | 29.918 |
| summarize_scenes | 16.251 |
| synthesize_synopsis | 29.677 |
| make_embedding | 4.058 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.816 |
| branch_yolo_total | 10.926 |
| branch_audio_total | 51.453 |

## 2026-06-27 14:54:24 UTC | 1IDUWll4TXo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1IDUWll4TXo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.718` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.574 |
| caption_frames | 44.172 |
| sample_fps | 2.389 |
| detect_object_yolo | 8.500 |
| audio_scan | 11.783 |
| asr_timings | 6.966 |
| ast_timings | 32.828 |
| describe_scenes | 8.476 |
| summarize_scenes | 11.914 |
| synthesize_synopsis | 9.028 |
| make_embedding | 3.900 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.751 |
| branch_yolo_total | 10.895 |
| branch_audio_total | 51.586 |
