# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:28:37 UTC | 0vQvjLp_b4w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.480 | 0.788 | 41.695 | 10.854 | 12.424 | 19.226 | 2.553 |
| 2026-06-27 14:18:55 UTC | 0vQvjLp_b4w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.290 | 0.789 | 42.379 | 5.711 | 7.820 | 10.458 | 3.090 |

## 2026-06-23 12:28:37 UTC | 0vQvjLp_b4w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0vQvjLp_b4w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.480` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 0.676 |
| caption_frames | 25.951 |
| sample_fps | 2.020 |
| detect_object_yolo | 6.925 |
| audio_scan | 14.721 |
| asr_timings | 9.166 |
| ast_timings | 17.799 |
| describe_scenes | 10.854 |
| summarize_scenes | 12.424 |
| synthesize_synopsis | 19.226 |
| make_embedding | 2.553 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.632 |
| branch_yolo_total | 8.951 |
| branch_audio_total | 41.695 |

## 2026-06-27 14:18:55 UTC | 0vQvjLp_b4w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0vQvjLp_b4w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.290` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 0.676 |
| caption_frames | 26.903 |
| sample_fps | 2.037 |
| detect_object_yolo | 7.013 |
| audio_scan | 14.979 |
| asr_timings | 9.168 |
| ast_timings | 18.223 |
| describe_scenes | 5.711 |
| summarize_scenes | 7.820 |
| synthesize_synopsis | 10.458 |
| make_embedding | 3.090 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.585 |
| branch_yolo_total | 9.056 |
| branch_audio_total | 42.379 |
