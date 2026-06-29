# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:55:37 UTC | 4ExyMj2938I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 212.951 | 0.798 | 61.574 | 27.501 | 35.986 | 15.739 | 5.059 |
| 2026-06-24 10:50:32 UTC | 4ExyMj2938I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 231.750 | 0.813 | 62.930 | 18.698 | 41.663 | 34.137 | 4.955 |

## 2026-06-23 16:55:37 UTC | 4ExyMj2938I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4ExyMj2938I_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `212.951` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 1.606 |
| caption_frames | 50.344 |
| sample_fps | 2.458 |
| detect_object_yolo | 10.519 |
| audio_scan | 10.606 |
| asr_timings | 10.245 |
| ast_timings | 40.716 |
| describe_scenes | 27.501 |
| summarize_scenes | 35.986 |
| synthesize_synopsis | 15.739 |
| make_embedding | 5.059 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.956 |
| branch_yolo_total | 12.982 |
| branch_audio_total | 61.574 |

## 2026-06-24 10:50:32 UTC | 4ExyMj2938I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4ExyMj2938I_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `231.750` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.813 |
| save_clips | - |
| sample_frames | 1.615 |
| caption_frames | 52.584 |
| sample_fps | 2.462 |
| detect_object_yolo | 10.514 |
| audio_scan | 10.657 |
| asr_timings | 11.551 |
| ast_timings | 40.714 |
| describe_scenes | 18.698 |
| summarize_scenes | 41.663 |
| synthesize_synopsis | 34.137 |
| make_embedding | 4.955 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.205 |
| branch_yolo_total | 12.981 |
| branch_audio_total | 62.930 |
