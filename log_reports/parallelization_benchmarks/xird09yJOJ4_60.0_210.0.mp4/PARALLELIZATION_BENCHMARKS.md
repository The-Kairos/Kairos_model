# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:01:27 UTC | xird09yJOJ4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 85.318 | 0.777 | 33.793 | 5.041 | 3.845 | 10.495 | 2.044 |

## 2026-06-27 04:01:27 UTC | xird09yJOJ4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xird09yJOJ4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `85.318` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 0.455 |
| caption_frames | 18.581 |
| sample_fps | 1.951 |
| detect_object_yolo | 6.941 |
| audio_scan | 11.963 |
| asr_timings | 9.116 |
| ast_timings | 12.705 |
| describe_scenes | 5.041 |
| summarize_scenes | 3.845 |
| synthesize_synopsis | 10.495 |
| make_embedding | 2.044 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.043 |
| branch_yolo_total | 8.898 |
| branch_audio_total | 33.793 |
