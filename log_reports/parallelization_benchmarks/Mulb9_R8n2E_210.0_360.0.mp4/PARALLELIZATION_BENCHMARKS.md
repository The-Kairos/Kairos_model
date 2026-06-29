# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:01:46 UTC | Mulb9_R8n2E_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.507 | 0.791 | 70.276 | 12.914 | 14.235 | 29.915 | 3.090 |

## 2026-06-25 10:01:46 UTC | Mulb9_R8n2E_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Mulb9_R8n2E_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.507` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 0.946 |
| caption_frames | 33.145 |
| sample_fps | 2.167 |
| detect_object_yolo | 8.615 |
| audio_scan | 14.854 |
| asr_timings | 31.407 |
| ast_timings | 24.006 |
| describe_scenes | 12.914 |
| summarize_scenes | 14.235 |
| synthesize_synopsis | 29.915 |
| make_embedding | 3.090 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.096 |
| branch_yolo_total | 10.788 |
| branch_audio_total | 70.276 |
