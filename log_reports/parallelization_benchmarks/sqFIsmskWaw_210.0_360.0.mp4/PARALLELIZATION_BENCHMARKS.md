# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 22:32:47 UTC | sqFIsmskWaw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.919 | 0.791 | 56.599 | 13.733 | 19.782 | 12.132 | 3.854 |

## 2026-06-26 22:32:47 UTC | sqFIsmskWaw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sqFIsmskWaw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.919` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.352 |
| caption_frames | 41.901 |
| sample_fps | 2.369 |
| detect_object_yolo | 8.994 |
| audio_scan | 13.919 |
| asr_timings | 9.942 |
| ast_timings | 32.729 |
| describe_scenes | 13.733 |
| summarize_scenes | 19.782 |
| synthesize_synopsis | 12.132 |
| make_embedding | 3.854 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.258 |
| branch_yolo_total | 11.369 |
| branch_audio_total | 56.599 |
