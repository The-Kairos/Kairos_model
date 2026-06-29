# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:48:59 UTC | 4ExyMj2938I_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.128 | 0.756 | 51.700 | 17.807 | 18.954 | 19.089 | 3.421 |
| 2026-06-24 10:43:59 UTC | 4ExyMj2938I_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.139 | 0.837 | 52.280 | 13.079 | 9.971 | 25.527 | 3.370 |

## 2026-06-23 16:48:59 UTC | 4ExyMj2938I_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4ExyMj2938I_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.128` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.756 |
| save_clips | - |
| sample_frames | 0.921 |
| caption_frames | 36.411 |
| sample_fps | 2.157 |
| detect_object_yolo | 8.524 |
| audio_scan | 14.802 |
| asr_timings | 10.272 |
| ast_timings | 26.618 |
| describe_scenes | 17.807 |
| summarize_scenes | 18.954 |
| synthesize_synopsis | 19.089 |
| make_embedding | 3.421 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.337 |
| branch_yolo_total | 10.686 |
| branch_audio_total | 51.700 |

## 2026-06-24 10:43:59 UTC | 4ExyMj2938I_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4ExyMj2938I_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.139` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.837 |
| save_clips | - |
| sample_frames | 0.946 |
| caption_frames | 36.925 |
| sample_fps | 2.264 |
| detect_object_yolo | 8.520 |
| audio_scan | 14.894 |
| asr_timings | 10.651 |
| ast_timings | 26.726 |
| describe_scenes | 13.079 |
| summarize_scenes | 9.971 |
| synthesize_synopsis | 25.527 |
| make_embedding | 3.370 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.876 |
| branch_yolo_total | 10.789 |
| branch_audio_total | 52.280 |
