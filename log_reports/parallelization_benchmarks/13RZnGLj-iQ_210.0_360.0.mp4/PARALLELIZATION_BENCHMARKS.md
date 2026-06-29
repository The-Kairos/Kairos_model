# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:39:13 UTC | 13RZnGLj-iQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.557 | 0.767 | 47.585 | 15.631 | 18.702 | 19.588 | 3.833 |
| 2026-06-27 14:26:07 UTC | 13RZnGLj-iQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.022 | 0.777 | 50.563 | 9.051 | 8.885 | 7.921 | 3.854 |

## 2026-06-23 12:39:13 UTC | 13RZnGLj-iQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13RZnGLj-iQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.557` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.122 |
| caption_frames | 40.634 |
| sample_fps | 2.277 |
| detect_object_yolo | 9.008 |
| audio_scan | 7.417 |
| asr_timings | 7.680 |
| ast_timings | 32.480 |
| describe_scenes | 15.631 |
| summarize_scenes | 18.702 |
| synthesize_synopsis | 19.588 |
| make_embedding | 3.833 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.762 |
| branch_yolo_total | 11.292 |
| branch_audio_total | 47.585 |

## 2026-06-27 14:26:07 UTC | 13RZnGLj-iQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13RZnGLj-iQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.022` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 1.127 |
| caption_frames | 41.990 |
| sample_fps | 2.326 |
| detect_object_yolo | 9.107 |
| audio_scan | 7.505 |
| asr_timings | 10.292 |
| ast_timings | 32.757 |
| describe_scenes | 9.051 |
| summarize_scenes | 8.885 |
| synthesize_synopsis | 7.921 |
| make_embedding | 3.854 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.123 |
| branch_yolo_total | 11.439 |
| branch_audio_total | 50.563 |
