# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:28:58 UTC | 32K4f8sskdM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.064 | 0.643 | 70.638 | 26.270 | 38.289 | 24.029 | 3.905 |
| 2026-06-24 09:26:39 UTC | 32K4f8sskdM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.267 | 0.659 | 70.648 | 18.924 | 10.602 | 20.729 | 3.949 |

## 2026-06-23 15:28:58 UTC | 32K4f8sskdM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/32K4f8sskdM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.064` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 1.595 |
| caption_frames | 43.543 |
| sample_fps | 2.289 |
| detect_object_yolo | 9.469 |
| audio_scan | 13.770 |
| asr_timings | 24.771 |
| ast_timings | 32.088 |
| describe_scenes | 26.270 |
| summarize_scenes | 38.289 |
| synthesize_synopsis | 24.029 |
| make_embedding | 3.905 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.144 |
| branch_yolo_total | 11.763 |
| branch_audio_total | 70.638 |

## 2026-06-24 09:26:39 UTC | 32K4f8sskdM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/32K4f8sskdM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.267` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.659 |
| save_clips | - |
| sample_frames | 1.630 |
| caption_frames | 42.975 |
| sample_fps | 2.267 |
| detect_object_yolo | 9.488 |
| audio_scan | 13.849 |
| asr_timings | 24.666 |
| ast_timings | 32.124 |
| describe_scenes | 18.924 |
| summarize_scenes | 10.602 |
| synthesize_synopsis | 20.729 |
| make_embedding | 3.949 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.611 |
| branch_yolo_total | 11.761 |
| branch_audio_total | 70.648 |
