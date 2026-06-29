# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:41:00 UTC | b2RgDRFM55c_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 233.806 | 0.678 | 97.995 | 20.176 | 11.847 | 11.901 | 5.991 |

## 2026-06-26 00:41:00 UTC | b2RgDRFM55c_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/b2RgDRFM55c_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `233.806` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.678 |
| save_clips | - |
| sample_frames | 1.808 |
| caption_frames | 67.543 |
| sample_fps | 2.484 |
| detect_object_yolo | 11.957 |
| audio_scan | 16.070 |
| asr_timings | 31.757 |
| ast_timings | 50.160 |
| describe_scenes | 20.176 |
| summarize_scenes | 11.847 |
| synthesize_synopsis | 11.901 |
| make_embedding | 5.991 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 69.357 |
| branch_yolo_total | 14.447 |
| branch_audio_total | 97.995 |
