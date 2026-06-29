# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:57:30 UTC | 1A6qJhSsuWc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 320.375 | 0.785 | 82.175 | 37.998 | 65.870 | 32.518 | 7.217 |
| 2026-06-27 14:40:52 UTC | 1A6qJhSsuWc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 226.702 | 0.772 | 83.731 | 17.639 | 14.353 | 7.989 | 6.516 |

## 2026-06-23 12:57:30 UTC | 1A6qJhSsuWc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1A6qJhSsuWc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `320.375` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.721 |
| caption_frames | 75.279 |
| sample_fps | 2.673 |
| detect_object_yolo | 12.773 |
| audio_scan | 14.705 |
| asr_timings | 14.203 |
| ast_timings | 53.259 |
| describe_scenes | 37.998 |
| summarize_scenes | 65.870 |
| synthesize_synopsis | 32.518 |
| make_embedding | 7.217 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 77.005 |
| branch_yolo_total | 15.451 |
| branch_audio_total | 82.175 |

## 2026-06-27 14:40:52 UTC | 1A6qJhSsuWc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1A6qJhSsuWc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `226.702` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 1.760 |
| caption_frames | 76.885 |
| sample_fps | 2.690 |
| detect_object_yolo | 12.942 |
| audio_scan | 14.835 |
| asr_timings | 14.495 |
| ast_timings | 54.393 |
| describe_scenes | 17.639 |
| summarize_scenes | 14.353 |
| synthesize_synopsis | 7.989 |
| make_embedding | 6.516 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 78.651 |
| branch_yolo_total | 15.638 |
| branch_audio_total | 83.731 |
