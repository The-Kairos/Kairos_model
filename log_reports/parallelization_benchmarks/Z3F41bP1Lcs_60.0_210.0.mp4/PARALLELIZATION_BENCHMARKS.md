# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:19:09 UTC | Z3F41bP1Lcs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.503 | 0.647 | 50.064 | 12.493 | 19.756 | 11.363 | 3.561 |

## 2026-06-25 22:19:09 UTC | Z3F41bP1Lcs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Z3F41bP1Lcs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.503` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.647 |
| save_clips | - |
| sample_frames | 0.900 |
| caption_frames | 41.068 |
| sample_fps | 2.089 |
| detect_object_yolo | 9.151 |
| audio_scan | 11.877 |
| asr_timings | 7.736 |
| ast_timings | 30.443 |
| describe_scenes | 12.493 |
| summarize_scenes | 19.756 |
| synthesize_synopsis | 11.363 |
| make_embedding | 3.561 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.974 |
| branch_yolo_total | 11.246 |
| branch_audio_total | 50.064 |
