# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:50:14 UTC | jogh5IF2rEY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.504 | 0.810 | 46.156 | 22.324 | 12.540 | 21.416 | 3.032 |

## 2026-06-26 11:50:14 UTC | jogh5IF2rEY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jogh5IF2rEY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.504` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 0.761 |
| caption_frames | 34.024 |
| sample_fps | 2.149 |
| detect_object_yolo | 8.848 |
| audio_scan | 12.924 |
| asr_timings | 9.187 |
| ast_timings | 24.036 |
| describe_scenes | 22.324 |
| summarize_scenes | 12.540 |
| synthesize_synopsis | 21.416 |
| make_embedding | 3.032 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.791 |
| branch_yolo_total | 11.003 |
| branch_audio_total | 46.156 |
