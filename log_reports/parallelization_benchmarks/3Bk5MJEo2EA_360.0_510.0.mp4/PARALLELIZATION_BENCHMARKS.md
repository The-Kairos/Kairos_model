# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:12:27 UTC | 3Bk5MJEo2EA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.911 | 1.763 | 45.764 | 13.080 | 4.817 | 8.944 | 3.296 |
| 2026-06-21 21:51:45 UTC | 3Bk5MJEo2EA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.150 | 1.726 | 46.316 | 7.622 | 5.060 | 7.327 | 3.262 |

## 2026-06-21 10:12:27 UTC | 3Bk5MJEo2EA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.911` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.763 |
| save_clips | - |
| sample_frames | 2.825 |
| caption_frames | 33.544 |
| sample_fps | 5.447 |
| detect_object_yolo | 8.136 |
| audio_scan | 10.562 |
| asr_timings | 9.254 |
| ast_timings | 25.940 |
| describe_scenes | 13.080 |
| summarize_scenes | 4.817 |
| synthesize_synopsis | 8.944 |
| make_embedding | 3.296 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.375 |
| branch_yolo_total | 13.590 |
| branch_audio_total | 45.764 |

## 2026-06-21 21:51:45 UTC | 3Bk5MJEo2EA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.150` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.726 |
| save_clips | - |
| sample_frames | 2.868 |
| caption_frames | 35.594 |
| sample_fps | 5.531 |
| detect_object_yolo | 8.451 |
| audio_scan | 10.856 |
| asr_timings | 9.256 |
| ast_timings | 26.195 |
| describe_scenes | 7.622 |
| summarize_scenes | 5.060 |
| synthesize_synopsis | 7.327 |
| make_embedding | 3.262 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.468 |
| branch_yolo_total | 13.989 |
| branch_audio_total | 46.316 |
