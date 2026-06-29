# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:54:03 UTC | 0zDV_cVzPiI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 13:05:45 UTC | 0zDV_cVzPiI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.692 | 0.638 | 40.141 | 7.688 | 17.658 | 22.595 | 2.169 |

## 2026-06-21 20:54:03 UTC | 0zDV_cVzPiI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0zDV_cVzPiI_60.0_210.0.mp4`
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

## 2026-06-22 13:05:45 UTC | 0zDV_cVzPiI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0zDV_cVzPiI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.692` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.638 |
| save_clips | - |
| sample_frames | 0.249 |
| caption_frames | 13.208 |
| sample_fps | 1.681 |
| detect_object_yolo | 6.270 |
| audio_scan | 14.917 |
| asr_timings | 15.250 |
| ast_timings | 9.965 |
| describe_scenes | 7.688 |
| summarize_scenes | 17.658 |
| synthesize_synopsis | 22.595 |
| make_embedding | 2.169 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.463 |
| branch_yolo_total | 7.957 |
| branch_audio_total | 40.141 |
