# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:25:19 UTC | 0HAACVba7kI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 120.484 | 0.889 | 51.836 | 6.915 | 5.223 | 13.288 | 2.782 |

## 2026-06-27 13:25:19 UTC | 0HAACVba7kI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0HAACVba7kI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `120.484` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.889 |
| save_clips | - |
| sample_frames | 0.903 |
| caption_frames | 27.914 |
| sample_fps | 2.122 |
| detect_object_yolo | 7.220 |
| audio_scan | 15.880 |
| asr_timings | 14.899 |
| ast_timings | 21.048 |
| describe_scenes | 6.915 |
| summarize_scenes | 5.223 |
| synthesize_synopsis | 13.288 |
| make_embedding | 2.782 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.823 |
| branch_yolo_total | 9.348 |
| branch_audio_total | 51.836 |
