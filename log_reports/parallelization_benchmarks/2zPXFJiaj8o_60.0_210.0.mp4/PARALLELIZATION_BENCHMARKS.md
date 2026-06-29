# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:04:04 UTC | 2zPXFJiaj8o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.017 | 0.648 | 50.038 | 26.904 | 41.930 | 19.225 | 3.119 |
| 2026-06-24 09:04:13 UTC | 2zPXFJiaj8o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.897 | 0.671 | 50.430 | 19.450 | 19.678 | 18.063 | 3.202 |

## 2026-06-23 15:04:04 UTC | 2zPXFJiaj8o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zPXFJiaj8o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.017` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.648 |
| save_clips | - |
| sample_frames | 0.876 |
| caption_frames | 32.903 |
| sample_fps | 1.988 |
| detect_object_yolo | 8.017 |
| audio_scan | 14.794 |
| asr_timings | 11.338 |
| ast_timings | 23.898 |
| describe_scenes | 26.904 |
| summarize_scenes | 41.930 |
| synthesize_synopsis | 19.225 |
| make_embedding | 3.119 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.784 |
| branch_yolo_total | 10.011 |
| branch_audio_total | 50.038 |

## 2026-06-24 09:04:13 UTC | 2zPXFJiaj8o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zPXFJiaj8o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.897` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 0.894 |
| caption_frames | 33.921 |
| sample_fps | 2.035 |
| detect_object_yolo | 8.136 |
| audio_scan | 14.877 |
| asr_timings | 11.628 |
| ast_timings | 23.917 |
| describe_scenes | 19.450 |
| summarize_scenes | 19.678 |
| synthesize_synopsis | 18.063 |
| make_embedding | 3.202 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.821 |
| branch_yolo_total | 10.177 |
| branch_audio_total | 50.430 |
