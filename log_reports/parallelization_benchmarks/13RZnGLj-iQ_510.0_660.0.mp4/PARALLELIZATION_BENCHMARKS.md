# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:43:27 UTC | 13RZnGLj-iQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 129.031 | 0.772 | 44.571 | 13.273 | 11.388 | 17.453 | 2.837 |
| 2026-06-27 14:29:40 UTC | 13RZnGLj-iQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.501 | 0.793 | 45.908 | 5.965 | 13.231 | 5.953 | 2.814 |

## 2026-06-23 12:43:27 UTC | 13RZnGLj-iQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13RZnGLj-iQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `129.031` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 0.801 |
| caption_frames | 27.134 |
| sample_fps | 2.035 |
| detect_object_yolo | 7.388 |
| audio_scan | 13.598 |
| asr_timings | 10.220 |
| ast_timings | 20.745 |
| describe_scenes | 13.273 |
| summarize_scenes | 11.388 |
| synthesize_synopsis | 17.453 |
| make_embedding | 2.837 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.941 |
| branch_yolo_total | 9.428 |
| branch_audio_total | 44.571 |

## 2026-06-27 14:29:40 UTC | 13RZnGLj-iQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13RZnGLj-iQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.501` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 0.821 |
| caption_frames | 28.131 |
| sample_fps | 2.058 |
| detect_object_yolo | 7.422 |
| audio_scan | 13.867 |
| asr_timings | 10.991 |
| ast_timings | 21.042 |
| describe_scenes | 5.965 |
| summarize_scenes | 13.231 |
| synthesize_synopsis | 5.953 |
| make_embedding | 2.814 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.958 |
| branch_yolo_total | 9.485 |
| branch_audio_total | 45.908 |
