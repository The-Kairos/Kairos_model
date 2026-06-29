# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:57:37 UTC | 4U-GkU4Vryo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.201 | 0.653 | 39.897 | 12.563 | 15.038 | 16.806 | 2.329 |
| 2026-06-24 10:52:24 UTC | 4U-GkU4Vryo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.830 | 0.655 | 40.642 | 8.418 | 6.344 | 21.207 | 2.368 |

## 2026-06-23 16:57:37 UTC | 4U-GkU4Vryo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4U-GkU4Vryo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.201` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.653 |
| save_clips | - |
| sample_frames | 0.464 |
| caption_frames | 20.246 |
| sample_fps | 1.809 |
| detect_object_yolo | 7.024 |
| audio_scan | 11.640 |
| asr_timings | 12.608 |
| ast_timings | 15.640 |
| describe_scenes | 12.563 |
| summarize_scenes | 15.038 |
| synthesize_synopsis | 16.806 |
| make_embedding | 2.329 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.715 |
| branch_yolo_total | 8.839 |
| branch_audio_total | 39.897 |

## 2026-06-24 10:52:24 UTC | 4U-GkU4Vryo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4U-GkU4Vryo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.830` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.655 |
| save_clips | - |
| sample_frames | 0.461 |
| caption_frames | 20.395 |
| sample_fps | 1.822 |
| detect_object_yolo | 7.138 |
| audio_scan | 11.658 |
| asr_timings | 13.305 |
| ast_timings | 15.671 |
| describe_scenes | 8.418 |
| summarize_scenes | 6.344 |
| synthesize_synopsis | 21.207 |
| make_embedding | 2.368 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.862 |
| branch_yolo_total | 8.966 |
| branch_audio_total | 40.642 |
