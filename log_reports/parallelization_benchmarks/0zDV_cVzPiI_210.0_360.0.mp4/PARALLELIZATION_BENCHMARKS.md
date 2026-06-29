# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:54:01 UTC | 0zDV_cVzPiI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | - |
| 2026-06-22 13:01:19 UTC | 0zDV_cVzPiI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 190.807 | 0.654 | 59.796 | 27.438 | 24.070 | 24.901 | 3.553 |

## 2026-06-21 20:54:01 UTC | 0zDV_cVzPiI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0zDV_cVzPiI_210.0_360.0.mp4`
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

## 2026-06-22 13:01:19 UTC | 0zDV_cVzPiI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0zDV_cVzPiI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.807` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.654 |
| save_clips | - |
| sample_frames | 1.052 |
| caption_frames | 37.233 |
| sample_fps | 2.084 |
| detect_object_yolo | 8.654 |
| audio_scan | 12.763 |
| asr_timings | 17.361 |
| ast_timings | 29.664 |
| describe_scenes | 27.438 |
| summarize_scenes | 24.070 |
| synthesize_synopsis | 24.901 |
| make_embedding | 3.553 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.291 |
| branch_yolo_total | 10.744 |
| branch_audio_total | 59.796 |
