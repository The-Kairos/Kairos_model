# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:53:50 UTC | P54sP0Nlngg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 247.239 | 0.797 | 73.277 | 32.660 | 22.926 | 24.498 | 6.549 |

## 2026-06-25 12:53:50 UTC | P54sP0Nlngg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/P54sP0Nlngg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `247.239` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 1.558 |
| caption_frames | 68.627 |
| sample_fps | 2.614 |
| detect_object_yolo | 12.309 |
| audio_scan | 12.157 |
| asr_timings | 11.626 |
| ast_timings | 49.485 |
| describe_scenes | 32.660 |
| summarize_scenes | 22.926 |
| synthesize_synopsis | 24.498 |
| make_embedding | 6.549 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 70.191 |
| branch_yolo_total | 14.929 |
| branch_audio_total | 73.277 |
