# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:25:42 UTC | 2BzoQ31IPhk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.463 | 1.546 | 51.919 | 8.254 | 7.292 | 7.665 | 3.073 |
| 2026-06-21 20:54:15 UTC | 2BzoQ31IPhk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | - |
| 2026-06-22 13:40:46 UTC | 2BzoQ31IPhk_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.701 | 1.584 | 52.346 | 22.528 | 34.600 | 26.148 | 3.039 |

## 2026-06-21 09:25:42 UTC | 2BzoQ31IPhk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.463` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.546 |
| save_clips | - |
| sample_frames | 1.829 |
| caption_frames | 30.122 |
| sample_fps | 5.679 |
| detect_object_yolo | 7.789 |
| audio_scan | 14.862 |
| asr_timings | 13.312 |
| ast_timings | 23.737 |
| describe_scenes | 8.254 |
| summarize_scenes | 7.292 |
| synthesize_synopsis | 7.665 |
| make_embedding | 3.073 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.957 |
| branch_yolo_total | 13.474 |
| branch_audio_total | 51.919 |

## 2026-06-21 20:54:15 UTC | 2BzoQ31IPhk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.061` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 13:40:46 UTC | 2BzoQ31IPhk_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2BzoQ31IPhk_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.701` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.584 |
| save_clips | - |
| sample_frames | 1.753 |
| caption_frames | 31.038 |
| sample_fps | 5.883 |
| detect_object_yolo | 8.328 |
| audio_scan | 15.175 |
| asr_timings | 13.048 |
| ast_timings | 24.113 |
| describe_scenes | 22.528 |
| summarize_scenes | 34.600 |
| synthesize_synopsis | 26.148 |
| make_embedding | 3.039 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.797 |
| branch_yolo_total | 14.218 |
| branch_audio_total | 52.346 |
