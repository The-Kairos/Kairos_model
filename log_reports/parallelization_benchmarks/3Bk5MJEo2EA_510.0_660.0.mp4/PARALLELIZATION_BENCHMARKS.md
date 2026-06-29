# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:15:08 UTC | 3Bk5MJEo2EA_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.812 | 1.735 | 56.254 | 9.500 | 15.770 | 8.817 | 3.963 |
| 2026-06-21 21:54:22 UTC | 3Bk5MJEo2EA_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.220 | 1.761 | 56.827 | 10.295 | 9.764 | 7.496 | 3.893 |

## 2026-06-21 10:15:08 UTC | 3Bk5MJEo2EA_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.812` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.735 |
| save_clips | - |
| sample_frames | 4.319 |
| caption_frames | 43.910 |
| sample_fps | 6.415 |
| detect_object_yolo | 8.813 |
| audio_scan | 15.986 |
| asr_timings | 8.475 |
| ast_timings | 31.784 |
| describe_scenes | 9.500 |
| summarize_scenes | 15.770 |
| synthesize_synopsis | 8.817 |
| make_embedding | 3.963 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.236 |
| branch_yolo_total | 15.233 |
| branch_audio_total | 56.254 |

## 2026-06-21 21:54:22 UTC | 3Bk5MJEo2EA_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.220` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.761 |
| save_clips | - |
| sample_frames | 4.351 |
| caption_frames | 43.756 |
| sample_fps | 6.461 |
| detect_object_yolo | 9.223 |
| audio_scan | 16.061 |
| asr_timings | 8.615 |
| ast_timings | 32.144 |
| describe_scenes | 10.295 |
| summarize_scenes | 9.764 |
| synthesize_synopsis | 7.496 |
| make_embedding | 3.893 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.113 |
| branch_yolo_total | 15.690 |
| branch_audio_total | 56.827 |
