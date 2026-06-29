# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 22:30:03 UTC | sk3p9-ynrNE_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 1990.696 | 0.780 | 1896.216 | 14.039 | 13.102 | 8.736 | 3.679 |

## 2026-06-26 22:30:03 UTC | sk3p9-ynrNE_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sk3p9-ynrNE_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1990.696` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.292 |
| caption_frames | 39.933 |
| sample_fps | 2.299 |
| detect_object_yolo | 9.193 |
| audio_scan | 15.028 |
| asr_timings | 1850.967 |
| ast_timings | 30.212 |
| describe_scenes | 14.039 |
| summarize_scenes | 13.102 |
| synthesize_synopsis | 8.736 |
| make_embedding | 3.679 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.231 |
| branch_yolo_total | 11.498 |
| branch_audio_total | 1896.216 |
