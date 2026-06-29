# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:28:33 UTC | 3r7kOP_nYNo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.285 | 0.688 | 75.419 | 13.928 | 12.854 | 6.967 | 6.673 |

## 2026-06-21 22:28:33 UTC | 3r7kOP_nYNo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3r7kOP_nYNo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.285` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.688 |
| save_clips | - |
| sample_frames | 1.847 |
| caption_frames | 73.567 |
| sample_fps | 2.621 |
| detect_object_yolo | 13.457 |
| audio_scan | 5.434 |
| asr_timings | 12.545 |
| ast_timings | 56.287 |
| describe_scenes | 13.928 |
| summarize_scenes | 12.854 |
| synthesize_synopsis | 6.967 |
| make_embedding | 6.673 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 75.419 |
| branch_yolo_total | 16.083 |
| branch_audio_total | 74.275 |
