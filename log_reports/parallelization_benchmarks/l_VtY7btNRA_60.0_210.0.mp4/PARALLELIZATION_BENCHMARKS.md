# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:45:11 UTC | l_VtY7btNRA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.821 | 0.800 | 57.922 | 13.689 | 7.155 | 18.662 | 3.866 |

## 2026-06-26 15:45:11 UTC | l_VtY7btNRA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l_VtY7btNRA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.821` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.224 |
| caption_frames | 43.955 |
| sample_fps | 2.370 |
| detect_object_yolo | 9.717 |
| audio_scan | 13.032 |
| asr_timings | 12.245 |
| ast_timings | 32.637 |
| describe_scenes | 13.689 |
| summarize_scenes | 7.155 |
| synthesize_synopsis | 18.662 |
| make_embedding | 3.866 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.185 |
| branch_yolo_total | 12.093 |
| branch_audio_total | 57.922 |
