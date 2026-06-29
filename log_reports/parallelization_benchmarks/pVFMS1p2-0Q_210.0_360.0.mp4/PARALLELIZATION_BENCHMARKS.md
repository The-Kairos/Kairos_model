# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:07:58 UTC | pVFMS1p2-0Q_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 121.526 | 0.763 | 55.194 | 8.838 | 8.133 | 9.051 | 2.529 |

## 2026-06-28 08:07:58 UTC | pVFMS1p2-0Q_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pVFMS1p2-0Q_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `121.526` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.763 |
| save_clips | - |
| sample_frames | 0.692 |
| caption_frames | 25.961 |
| sample_fps | 2.021 |
| detect_object_yolo | 6.958 |
| audio_scan | 14.848 |
| asr_timings | 21.636 |
| ast_timings | 18.701 |
| describe_scenes | 8.838 |
| summarize_scenes | 8.133 |
| synthesize_synopsis | 9.051 |
| make_embedding | 2.529 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.659 |
| branch_yolo_total | 8.985 |
| branch_audio_total | 55.194 |
