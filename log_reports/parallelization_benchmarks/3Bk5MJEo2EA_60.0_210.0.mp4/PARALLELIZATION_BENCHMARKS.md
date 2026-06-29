# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:17:54 UTC | 3Bk5MJEo2EA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.630 | 1.786 | 61.049 | 14.565 | 6.931 | 6.397 | 4.447 |
| 2026-06-21 21:57:10 UTC | 3Bk5MJEo2EA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 167.595 | 1.809 | 61.840 | 10.427 | 12.222 | 5.706 | 4.490 |

## 2026-06-21 10:17:54 UTC | 3Bk5MJEo2EA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.630` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.786 |
| save_clips | - |
| sample_frames | 4.919 |
| caption_frames | 46.817 |
| sample_fps | 6.733 |
| detect_object_yolo | 9.668 |
| audio_scan | 14.881 |
| asr_timings | 8.958 |
| ast_timings | 37.201 |
| describe_scenes | 14.565 |
| summarize_scenes | 6.931 |
| synthesize_synopsis | 6.397 |
| make_embedding | 4.447 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.742 |
| branch_yolo_total | 16.406 |
| branch_audio_total | 61.049 |

## 2026-06-21 21:57:10 UTC | 3Bk5MJEo2EA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `167.595` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.809 |
| save_clips | - |
| sample_frames | 4.907 |
| caption_frames | 47.906 |
| sample_fps | 6.821 |
| detect_object_yolo | 10.080 |
| audio_scan | 15.122 |
| asr_timings | 8.883 |
| ast_timings | 37.826 |
| describe_scenes | 10.427 |
| summarize_scenes | 12.222 |
| synthesize_synopsis | 5.706 |
| make_embedding | 4.490 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.820 |
| branch_yolo_total | 16.907 |
| branch_audio_total | 61.840 |
