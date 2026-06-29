# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:59:47 UTC | 1icyCzbxmmg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 253.743 | 0.684 | 50.276 | 35.590 | 68.094 | 46.385 | 3.365 |
| 2026-06-27 15:24:06 UTC | 1icyCzbxmmg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.138 | 0.684 | 50.703 | 9.924 | 8.149 | 9.093 | 3.267 |

## 2026-06-23 13:59:47 UTC | 1icyCzbxmmg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1icyCzbxmmg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `253.743` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 1.355 |
| caption_frames | 36.132 |
| sample_fps | 2.134 |
| detect_object_yolo | 8.371 |
| audio_scan | 14.789 |
| asr_timings | 9.027 |
| ast_timings | 26.451 |
| describe_scenes | 35.590 |
| summarize_scenes | 68.094 |
| synthesize_synopsis | 46.385 |
| make_embedding | 3.365 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.493 |
| branch_yolo_total | 10.511 |
| branch_audio_total | 50.276 |

## 2026-06-27 15:24:06 UTC | 1icyCzbxmmg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1icyCzbxmmg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.138` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 1.350 |
| caption_frames | 35.849 |
| sample_fps | 2.140 |
| detect_object_yolo | 8.556 |
| audio_scan | 14.821 |
| asr_timings | 8.757 |
| ast_timings | 27.117 |
| describe_scenes | 9.924 |
| summarize_scenes | 8.149 |
| synthesize_synopsis | 9.093 |
| make_embedding | 3.267 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.205 |
| branch_yolo_total | 10.702 |
| branch_audio_total | 50.703 |
