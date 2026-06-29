# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:28:35 UTC | 2BzoQ31IPhk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.404 | 1.649 | 37.550 | 6.087 | 5.288 | 5.647 | 2.327 |
| 2026-06-21 20:55:58 UTC | 2BzoQ31IPhk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 100.953 | 1.659 | 40.300 | 5.895 | 14.502 | 6.162 | 2.435 |
| 2026-06-22 13:44:55 UTC | 2BzoQ31IPhk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.863 | 1.615 | 37.979 | 18.632 | 12.137 | 22.000 | 2.282 |

## 2026-06-21 09:28:35 UTC | 2BzoQ31IPhk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.404` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.649 |
| save_clips | - |
| sample_frames | 1.046 |
| caption_frames | 21.368 |
| sample_fps | 5.152 |
| detect_object_yolo | 6.974 |
| audio_scan | 12.771 |
| asr_timings | 9.253 |
| ast_timings | 15.517 |
| describe_scenes | 6.087 |
| summarize_scenes | 5.288 |
| synthesize_synopsis | 5.647 |
| make_embedding | 2.327 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.419 |
| branch_yolo_total | 12.132 |
| branch_audio_total | 37.550 |

## 2026-06-21 20:55:58 UTC | 2BzoQ31IPhk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `100.953` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.659 |
| save_clips | - |
| sample_frames | 1.062 |
| caption_frames | 15.495 |
| sample_fps | 5.231 |
| detect_object_yolo | 6.905 |
| audio_scan | 14.270 |
| asr_timings | 10.405 |
| ast_timings | 15.617 |
| describe_scenes | 5.895 |
| summarize_scenes | 14.502 |
| synthesize_synopsis | 6.162 |
| make_embedding | 2.435 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.563 |
| branch_yolo_total | 12.141 |
| branch_audio_total | 40.300 |

## 2026-06-22 13:44:55 UTC | 2BzoQ31IPhk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.863` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.615 |
| save_clips | - |
| sample_frames | 1.057 |
| caption_frames | 22.244 |
| sample_fps | 5.258 |
| detect_object_yolo | 7.256 |
| audio_scan | 12.902 |
| asr_timings | 9.333 |
| ast_timings | 15.736 |
| describe_scenes | 18.632 |
| summarize_scenes | 12.137 |
| synthesize_synopsis | 22.000 |
| make_embedding | 2.282 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.306 |
| branch_yolo_total | 12.521 |
| branch_audio_total | 37.979 |
