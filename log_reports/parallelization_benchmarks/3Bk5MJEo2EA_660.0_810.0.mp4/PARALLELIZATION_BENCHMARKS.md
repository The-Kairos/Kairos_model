# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:20:03 UTC | 3Bk5MJEo2EA_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 127.539 | 1.627 | 51.511 | 8.104 | 5.832 | 6.061 | 3.356 |
| 2026-06-21 21:59:27 UTC | 3Bk5MJEo2EA_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.517 | 1.607 | 52.720 | 8.517 | 8.709 | 7.801 | 3.394 |

## 2026-06-21 10:20:03 UTC | 3Bk5MJEo2EA_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.539` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.627 |
| save_clips | - |
| sample_frames | 2.547 |
| caption_frames | 33.907 |
| sample_fps | 5.409 |
| detect_object_yolo | 7.871 |
| audio_scan | 14.933 |
| asr_timings | 10.506 |
| ast_timings | 26.064 |
| describe_scenes | 8.104 |
| summarize_scenes | 5.832 |
| synthesize_synopsis | 6.061 |
| make_embedding | 3.356 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.460 |
| branch_yolo_total | 13.287 |
| branch_audio_total | 51.511 |

## 2026-06-21 21:59:27 UTC | 3Bk5MJEo2EA_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.517` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.607 |
| save_clips | - |
| sample_frames | 2.538 |
| caption_frames | 35.230 |
| sample_fps | 5.427 |
| detect_object_yolo | 8.190 |
| audio_scan | 15.052 |
| asr_timings | 11.242 |
| ast_timings | 26.417 |
| describe_scenes | 8.517 |
| summarize_scenes | 8.709 |
| synthesize_synopsis | 7.801 |
| make_embedding | 3.394 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.774 |
| branch_yolo_total | 13.623 |
| branch_audio_total | 52.720 |
