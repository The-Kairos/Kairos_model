# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:47:39 UTC | jna4ylDjww0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.955 | 0.680 | 54.043 | 19.749 | 23.024 | 18.216 | 3.642 |

## 2026-06-26 11:47:39 UTC | jna4ylDjww0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jna4ylDjww0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.955` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 0.907 |
| caption_frames | 41.873 |
| sample_fps | 2.080 |
| detect_object_yolo | 9.286 |
| audio_scan | 14.021 |
| asr_timings | 10.079 |
| ast_timings | 29.935 |
| describe_scenes | 19.749 |
| summarize_scenes | 23.024 |
| synthesize_synopsis | 18.216 |
| make_embedding | 3.642 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.786 |
| branch_yolo_total | 11.373 |
| branch_audio_total | 54.043 |
