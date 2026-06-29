# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:53:49 UTC | 3G2WCQnH6Nk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.660 | 0.766 | 43.144 | 15.288 | 11.158 | 35.540 | 1.844 |
| 2026-06-24 09:49:58 UTC | 3G2WCQnH6Nk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 105.720 | 0.803 | 41.876 | 5.158 | 13.222 | 18.544 | 1.822 |

## 2026-06-23 15:53:49 UTC | 3G2WCQnH6Nk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.660` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 0.380 |
| caption_frames | 14.664 |
| sample_fps | 1.880 |
| detect_object_yolo | 5.626 |
| audio_scan | 13.740 |
| asr_timings | 19.259 |
| ast_timings | 10.136 |
| describe_scenes | 15.288 |
| summarize_scenes | 11.158 |
| synthesize_synopsis | 35.540 |
| make_embedding | 1.844 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.050 |
| branch_yolo_total | 7.513 |
| branch_audio_total | 43.144 |

## 2026-06-24 09:49:58 UTC | 3G2WCQnH6Nk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3G2WCQnH6Nk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `105.720` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 0.378 |
| caption_frames | 15.046 |
| sample_fps | 1.874 |
| detect_object_yolo | 5.632 |
| audio_scan | 13.801 |
| asr_timings | 18.057 |
| ast_timings | 10.010 |
| describe_scenes | 5.158 |
| summarize_scenes | 13.222 |
| synthesize_synopsis | 18.544 |
| make_embedding | 1.822 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.429 |
| branch_yolo_total | 7.512 |
| branch_audio_total | 41.876 |
