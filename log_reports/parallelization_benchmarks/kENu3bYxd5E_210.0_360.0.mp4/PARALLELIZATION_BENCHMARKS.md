# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:29:04 UTC | kENu3bYxd5E_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 222.323 | 0.817 | 61.219 | 35.925 | 17.181 | 32.360 | 5.083 |

## 2026-06-26 13:29:04 UTC | kENu3bYxd5E_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kENu3bYxd5E_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `222.323` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.817 |
| save_clips | - |
| sample_frames | 1.539 |
| caption_frames | 53.735 |
| sample_fps | 2.490 |
| detect_object_yolo | 10.565 |
| audio_scan | 10.897 |
| asr_timings | 8.117 |
| ast_timings | 42.196 |
| describe_scenes | 35.925 |
| summarize_scenes | 17.181 |
| synthesize_synopsis | 32.360 |
| make_embedding | 5.083 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.279 |
| branch_yolo_total | 13.060 |
| branch_audio_total | 61.219 |
