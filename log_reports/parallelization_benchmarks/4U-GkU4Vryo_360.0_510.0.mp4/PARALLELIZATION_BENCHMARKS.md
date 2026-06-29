# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:01:41 UTC | 4U-GkU4Vryo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 243.168 | 0.635 | 66.032 | 23.890 | 52.840 | 25.711 | 5.047 |
| 2026-06-24 10:55:50 UTC | 4U-GkU4Vryo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 204.918 | 0.633 | 66.609 | 29.781 | 13.934 | 19.275 | 5.131 |

## 2026-06-23 17:01:41 UTC | 4U-GkU4Vryo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4U-GkU4Vryo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `243.168` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.635 |
| save_clips | - |
| sample_frames | 1.285 |
| caption_frames | 53.372 |
| sample_fps | 2.280 |
| detect_object_yolo | 10.709 |
| audio_scan | 15.896 |
| asr_timings | 9.496 |
| ast_timings | 40.632 |
| describe_scenes | 23.890 |
| summarize_scenes | 52.840 |
| synthesize_synopsis | 25.711 |
| make_embedding | 5.047 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.663 |
| branch_yolo_total | 12.994 |
| branch_audio_total | 66.032 |

## 2026-06-24 10:55:50 UTC | 4U-GkU4Vryo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4U-GkU4Vryo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `204.918` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.633 |
| save_clips | - |
| sample_frames | 1.294 |
| caption_frames | 53.727 |
| sample_fps | 2.284 |
| detect_object_yolo | 10.856 |
| audio_scan | 15.932 |
| asr_timings | 9.810 |
| ast_timings | 40.859 |
| describe_scenes | 29.781 |
| summarize_scenes | 13.934 |
| synthesize_synopsis | 19.275 |
| make_embedding | 5.131 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.027 |
| branch_yolo_total | 13.146 |
| branch_audio_total | 66.609 |
