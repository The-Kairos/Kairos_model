# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:08:39 UTC | kmJG4kjQXL8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.427 | 0.731 | 50.838 | 27.108 | 15.105 | 20.864 | 4.129 |

## 2026-06-26 14:08:39 UTC | kmJG4kjQXL8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kmJG4kjQXL8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.427` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.731 |
| save_clips | - |
| sample_frames | 1.107 |
| caption_frames | 45.415 |
| sample_fps | 2.202 |
| detect_object_yolo | 9.509 |
| audio_scan | 6.281 |
| asr_timings | 8.357 |
| ast_timings | 36.191 |
| describe_scenes | 27.108 |
| summarize_scenes | 15.105 |
| synthesize_synopsis | 20.864 |
| make_embedding | 4.129 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.528 |
| branch_yolo_total | 11.717 |
| branch_audio_total | 50.838 |
