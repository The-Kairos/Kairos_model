# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:00:49 UTC | Ys290kErJzE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.267 | 0.631 | 87.903 | 11.800 | 13.619 | 9.624 | 3.578 |

## 2026-06-25 22:00:49 UTC | Ys290kErJzE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ys290kErJzE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.267` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.631 |
| save_clips | - |
| sample_frames | 1.030 |
| caption_frames | 40.600 |
| sample_fps | 2.131 |
| detect_object_yolo | 8.936 |
| audio_scan | 9.666 |
| asr_timings | 47.617 |
| ast_timings | 30.612 |
| describe_scenes | 11.800 |
| summarize_scenes | 13.619 |
| synthesize_synopsis | 9.624 |
| make_embedding | 3.578 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.636 |
| branch_yolo_total | 11.072 |
| branch_audio_total | 87.903 |
