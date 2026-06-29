# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:59:04 UTC | 2zPXFJiaj8o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.528 | 0.640 | 35.538 | 8.343 | 24.955 | 23.332 | 2.081 |
| 2026-06-24 08:59:36 UTC | 2zPXFJiaj8o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.358 | 0.643 | 35.823 | 9.622 | 13.690 | 24.679 | 2.112 |

## 2026-06-23 14:59:04 UTC | 2zPXFJiaj8o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zPXFJiaj8o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.528` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.640 |
| save_clips | - |
| sample_frames | 0.342 |
| caption_frames | 19.173 |
| sample_fps | 1.738 |
| detect_object_yolo | 7.009 |
| audio_scan | 11.634 |
| asr_timings | 11.095 |
| ast_timings | 12.801 |
| describe_scenes | 8.343 |
| summarize_scenes | 24.955 |
| synthesize_synopsis | 23.332 |
| make_embedding | 2.081 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.521 |
| branch_yolo_total | 8.752 |
| branch_audio_total | 35.538 |

## 2026-06-24 08:59:36 UTC | 2zPXFJiaj8o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zPXFJiaj8o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.358` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 0.344 |
| caption_frames | 18.154 |
| sample_fps | 1.769 |
| detect_object_yolo | 7.112 |
| audio_scan | 11.736 |
| asr_timings | 11.216 |
| ast_timings | 12.862 |
| describe_scenes | 9.622 |
| summarize_scenes | 13.690 |
| synthesize_synopsis | 24.679 |
| make_embedding | 2.112 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.503 |
| branch_yolo_total | 8.887 |
| branch_audio_total | 35.823 |
