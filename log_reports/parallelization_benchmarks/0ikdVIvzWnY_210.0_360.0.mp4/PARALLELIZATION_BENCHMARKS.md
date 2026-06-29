# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 13:46:03 UTC | 0ikdVIvzWnY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.953 | 0.768 | 50.332 | 9.314 | 7.907 | 10.303 | 2.819 |

## 2026-06-27 13:46:03 UTC | 0ikdVIvzWnY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0ikdVIvzWnY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.953` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.768 |
| save_clips | - |
| sample_frames | 0.703 |
| caption_frames | 27.457 |
| sample_fps | 2.047 |
| detect_object_yolo | 6.906 |
| audio_scan | 14.903 |
| asr_timings | 13.995 |
| ast_timings | 21.425 |
| describe_scenes | 9.314 |
| summarize_scenes | 7.907 |
| synthesize_synopsis | 10.303 |
| make_embedding | 2.819 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.166 |
| branch_yolo_total | 8.960 |
| branch_audio_total | 50.332 |
