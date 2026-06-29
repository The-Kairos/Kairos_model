# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:54:02 UTC | 0zDV_cVzPiI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 13:03:50 UTC | 0zDV_cVzPiI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.459 | 0.640 | 48.904 | 18.173 | 14.917 | 21.142 | 3.349 |

## 2026-06-21 20:54:02 UTC | 0zDV_cVzPiI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0zDV_cVzPiI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.059` sec

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

## 2026-06-22 13:03:50 UTC | 0zDV_cVzPiI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0zDV_cVzPiI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.459` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 0.764 |
| caption_frames | 31.162 |
| sample_fps | 1.958 |
| detect_object_yolo | 8.066 |
| audio_scan | 14.829 |
| asr_timings | 10.032 |
| ast_timings | 24.035 |
| describe_scenes | 18.173 |
| summarize_scenes | 14.917 |
| synthesize_synopsis | 21.142 |
| make_embedding | 3.349 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.932 |
| branch_yolo_total | 10.029 |
| branch_audio_total | 48.904 |
