# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:56:58 UTC | 2boeKBw9x84_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 213.340 | 0.659 | 59.931 | 32.007 | 24.722 | 29.784 | 4.240 |
| 2026-06-24 08:57:39 UTC | 2boeKBw9x84_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.699 | 0.666 | 59.083 | 23.126 | 23.757 | 21.835 | 4.224 |

## 2026-06-23 14:56:58 UTC | 2boeKBw9x84_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boeKBw9x84_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `213.340` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.659 |
| save_clips | - |
| sample_frames | 1.716 |
| caption_frames | 46.695 |
| sample_fps | 2.371 |
| detect_object_yolo | 9.846 |
| audio_scan | 14.666 |
| asr_timings | 9.912 |
| ast_timings | 35.345 |
| describe_scenes | 32.007 |
| summarize_scenes | 24.722 |
| synthesize_synopsis | 29.784 |
| make_embedding | 4.240 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.417 |
| branch_yolo_total | 12.223 |
| branch_audio_total | 59.931 |

## 2026-06-24 08:57:39 UTC | 2boeKBw9x84_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boeKBw9x84_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.699` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.666 |
| save_clips | - |
| sample_frames | 1.724 |
| caption_frames | 49.478 |
| sample_fps | 2.405 |
| detect_object_yolo | 10.003 |
| audio_scan | 14.933 |
| asr_timings | 8.907 |
| ast_timings | 35.235 |
| describe_scenes | 23.126 |
| summarize_scenes | 23.757 |
| synthesize_synopsis | 21.835 |
| make_embedding | 4.224 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.208 |
| branch_yolo_total | 12.414 |
| branch_audio_total | 59.083 |
