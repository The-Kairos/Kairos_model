# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:44:56 UTC | b2RgDRFM55c_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 235.101 | 0.679 | 94.732 | 20.281 | 10.235 | 12.595 | 6.379 |

## 2026-06-26 00:44:56 UTC | b2RgDRFM55c_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/b2RgDRFM55c_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `235.101` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.679 |
| save_clips | - |
| sample_frames | 2.172 |
| caption_frames | 71.548 |
| sample_fps | 2.614 |
| detect_object_yolo | 12.447 |
| audio_scan | 11.622 |
| asr_timings | 31.153 |
| ast_timings | 51.949 |
| describe_scenes | 20.281 |
| summarize_scenes | 10.235 |
| synthesize_synopsis | 12.595 |
| make_embedding | 6.379 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 73.726 |
| branch_yolo_total | 15.066 |
| branch_audio_total | 94.732 |
