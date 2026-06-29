# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:49:28 UTC | VayyLoioSAk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.691 | 0.634 | 63.925 | 18.147 | 13.476 | 11.092 | 4.389 |

## 2026-06-25 19:49:28 UTC | VayyLoioSAk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VayyLoioSAk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.691` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.634 |
| save_clips | - |
| sample_frames | 1.107 |
| caption_frames | 48.309 |
| sample_fps | 2.138 |
| detect_object_yolo | 10.082 |
| audio_scan | 15.936 |
| asr_timings | 10.186 |
| ast_timings | 37.794 |
| describe_scenes | 18.147 |
| summarize_scenes | 13.476 |
| synthesize_synopsis | 11.092 |
| make_embedding | 4.389 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.422 |
| branch_yolo_total | 12.226 |
| branch_audio_total | 63.925 |
