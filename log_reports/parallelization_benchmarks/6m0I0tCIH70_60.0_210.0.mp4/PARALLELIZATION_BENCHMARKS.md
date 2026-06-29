# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:52:14 UTC | 6m0I0tCIH70_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.543 | 0.818 | 59.660 | 24.291 | 15.488 | 22.094 | 4.504 |

## 2026-06-24 12:52:14 UTC | 6m0I0tCIH70_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6m0I0tCIH70_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.543` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 1.418 |
| caption_frames | 49.424 |
| sample_fps | 2.409 |
| detect_object_yolo | 10.026 |
| audio_scan | 13.791 |
| asr_timings | 7.678 |
| ast_timings | 38.182 |
| describe_scenes | 24.291 |
| summarize_scenes | 15.488 |
| synthesize_synopsis | 22.094 |
| make_embedding | 4.504 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.848 |
| branch_yolo_total | 12.441 |
| branch_audio_total | 59.660 |
