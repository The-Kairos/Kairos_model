# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:55:37 UTC | jS3n3tcToHs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.829 | 0.801 | 47.478 | 30.165 | 26.373 | 16.180 | 3.336 |

## 2026-06-26 10:55:37 UTC | jS3n3tcToHs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jS3n3tcToHs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.829` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 0.935 |
| caption_frames | 35.470 |
| sample_fps | 2.155 |
| detect_object_yolo | 8.514 |
| audio_scan | 10.853 |
| asr_timings | 10.122 |
| ast_timings | 26.494 |
| describe_scenes | 30.165 |
| summarize_scenes | 26.373 |
| synthesize_synopsis | 16.180 |
| make_embedding | 3.336 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.411 |
| branch_yolo_total | 10.674 |
| branch_audio_total | 47.478 |
