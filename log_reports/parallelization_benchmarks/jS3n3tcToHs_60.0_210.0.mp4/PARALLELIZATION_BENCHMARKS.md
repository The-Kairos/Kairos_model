# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:03:57 UTC | jS3n3tcToHs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.688 | 0.791 | 54.394 | 22.110 | 15.187 | 41.119 | 3.084 |

## 2026-06-26 11:03:57 UTC | jS3n3tcToHs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jS3n3tcToHs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.688` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.000 |
| caption_frames | 33.107 |
| sample_fps | 2.151 |
| detect_object_yolo | 8.324 |
| audio_scan | 13.988 |
| asr_timings | 15.905 |
| ast_timings | 24.493 |
| describe_scenes | 22.110 |
| summarize_scenes | 15.187 |
| synthesize_synopsis | 41.119 |
| make_embedding | 3.084 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.113 |
| branch_yolo_total | 10.481 |
| branch_audio_total | 54.394 |
