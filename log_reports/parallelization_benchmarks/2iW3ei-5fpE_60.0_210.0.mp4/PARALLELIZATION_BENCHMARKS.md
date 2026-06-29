# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:55:14 UTC | 2iW3ei-5fpE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 120.065 | 1.399 | 45.331 | 5.754 | 7.232 | 7.601 | 3.073 |
| 2026-06-21 21:34:40 UTC | 2iW3ei-5fpE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.345 | 1.455 | 45.734 | 6.474 | 8.124 | 8.685 | 3.018 |

## 2026-06-21 09:55:14 UTC | 2iW3ei-5fpE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2iW3ei-5fpE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `120.065` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.399 |
| save_clips | - |
| sample_frames | 1.862 |
| caption_frames | 33.356 |
| sample_fps | 5.057 |
| detect_object_yolo | 8.064 |
| audio_scan | 11.788 |
| asr_timings | 9.774 |
| ast_timings | 23.762 |
| describe_scenes | 5.754 |
| summarize_scenes | 7.232 |
| synthesize_synopsis | 7.601 |
| make_embedding | 3.073 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.224 |
| branch_yolo_total | 13.127 |
| branch_audio_total | 45.331 |

## 2026-06-21 21:34:40 UTC | 2iW3ei-5fpE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2iW3ei-5fpE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.345` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.455 |
| save_clips | - |
| sample_frames | 1.873 |
| caption_frames | 34.299 |
| sample_fps | 5.093 |
| detect_object_yolo | 8.193 |
| audio_scan | 11.832 |
| asr_timings | 9.720 |
| ast_timings | 24.174 |
| describe_scenes | 6.474 |
| summarize_scenes | 8.124 |
| synthesize_synopsis | 8.685 |
| make_embedding | 3.018 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.177 |
| branch_yolo_total | 13.292 |
| branch_audio_total | 45.734 |
