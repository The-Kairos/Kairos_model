# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:27:04 UTC | r62uMwPmBK0_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.050 | 0.657 | 39.703 | 17.774 | 11.716 | 20.895 | 3.481 |

## 2026-06-26 18:27:04 UTC | r62uMwPmBK0_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/r62uMwPmBK0_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.050` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 1.317 |
| caption_frames | 38.381 |
| sample_fps | 2.223 |
| detect_object_yolo | 9.163 |
| audio_scan | 1.071 |
| asr_timings | 0.000 |
| ast_timings | 0.000 |
| describe_scenes | 17.774 |
| summarize_scenes | 11.716 |
| synthesize_synopsis | 20.895 |
| make_embedding | 3.481 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.703 |
| branch_yolo_total | 11.392 |
| branch_audio_total | 1.079 |
