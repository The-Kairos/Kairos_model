# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:24:02 UTC | CXDlWaDi7a0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 121.798 | 0.691 | 45.396 | 6.513 | 10.669 | 13.937 | 2.749 |

## 2026-06-24 20:24:02 UTC | CXDlWaDi7a0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CXDlWaDi7a0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `121.798` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.691 |
| save_clips | - |
| sample_frames | 0.782 |
| caption_frames | 30.144 |
| sample_fps | 1.931 |
| detect_object_yolo | 7.560 |
| audio_scan | 15.088 |
| asr_timings | 9.402 |
| ast_timings | 20.898 |
| describe_scenes | 6.513 |
| summarize_scenes | 10.669 |
| synthesize_synopsis | 13.937 |
| make_embedding | 2.749 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.931 |
| branch_yolo_total | 9.497 |
| branch_audio_total | 45.396 |
