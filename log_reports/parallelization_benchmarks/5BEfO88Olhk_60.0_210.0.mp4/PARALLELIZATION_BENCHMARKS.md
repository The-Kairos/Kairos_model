# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:36:59 UTC | 5BEfO88Olhk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 211.842 | 0.775 | 52.320 | 23.128 | 51.365 | 19.341 | 4.280 |
| 2026-06-24 11:29:18 UTC | 5BEfO88Olhk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.779 | 0.776 | 52.221 | 15.872 | 20.747 | 18.460 | 4.363 |

## 2026-06-23 17:36:59 UTC | 5BEfO88Olhk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5BEfO88Olhk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `211.842` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 1.108 |
| caption_frames | 46.210 |
| sample_fps | 2.281 |
| detect_object_yolo | 9.649 |
| audio_scan | 7.443 |
| asr_timings | 9.020 |
| ast_timings | 35.849 |
| describe_scenes | 23.128 |
| summarize_scenes | 51.365 |
| synthesize_synopsis | 19.341 |
| make_embedding | 4.280 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.324 |
| branch_yolo_total | 11.936 |
| branch_audio_total | 52.320 |

## 2026-06-24 11:29:18 UTC | 5BEfO88Olhk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5BEfO88Olhk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.779` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 1.135 |
| caption_frames | 46.830 |
| sample_fps | 2.302 |
| detect_object_yolo | 9.659 |
| audio_scan | 7.502 |
| asr_timings | 9.019 |
| ast_timings | 35.692 |
| describe_scenes | 15.872 |
| summarize_scenes | 20.747 |
| synthesize_synopsis | 18.460 |
| make_embedding | 4.363 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.971 |
| branch_yolo_total | 11.967 |
| branch_audio_total | 52.221 |
