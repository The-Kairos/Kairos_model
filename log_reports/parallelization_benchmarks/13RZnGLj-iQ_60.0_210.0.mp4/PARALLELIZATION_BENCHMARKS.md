# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:44:59 UTC | 13RZnGLj-iQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 90.227 | 0.769 | 29.406 | 6.709 | 9.574 | 17.907 | 1.920 |
| 2026-06-27 14:30:56 UTC | 13RZnGLj-iQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 74.535 | 0.775 | 29.096 | 4.134 | 4.430 | 9.515 | 1.796 |

## 2026-06-23 12:44:59 UTC | 13RZnGLj-iQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13RZnGLj-iQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `90.227` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 0.319 |
| caption_frames | 14.579 |
| sample_fps | 1.854 |
| detect_object_yolo | 5.815 |
| audio_scan | 11.672 |
| asr_timings | 7.834 |
| ast_timings | 9.892 |
| describe_scenes | 6.709 |
| summarize_scenes | 9.574 |
| synthesize_synopsis | 17.907 |
| make_embedding | 1.920 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.904 |
| branch_yolo_total | 7.675 |
| branch_audio_total | 29.406 |

## 2026-06-27 14:30:56 UTC | 13RZnGLj-iQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13RZnGLj-iQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `74.535` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 0.321 |
| caption_frames | 15.193 |
| sample_fps | 1.890 |
| detect_object_yolo | 5.985 |
| audio_scan | 11.799 |
| asr_timings | 7.191 |
| ast_timings | 10.097 |
| describe_scenes | 4.134 |
| summarize_scenes | 4.430 |
| synthesize_synopsis | 9.515 |
| make_embedding | 1.796 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.519 |
| branch_yolo_total | 7.880 |
| branch_audio_total | 29.096 |
