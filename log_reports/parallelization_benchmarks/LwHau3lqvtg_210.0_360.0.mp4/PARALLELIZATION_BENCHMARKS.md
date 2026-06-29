# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:51:44 UTC | LwHau3lqvtg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 206.406 | 0.882 | 82.910 | 16.939 | 19.789 | 22.726 | 3.900 |

## 2026-06-25 07:51:44 UTC | LwHau3lqvtg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LwHau3lqvtg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `206.406` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.882 |
| save_clips | - |
| sample_frames | 1.155 |
| caption_frames | 45.025 |
| sample_fps | 2.324 |
| detect_object_yolo | 9.326 |
| audio_scan | 14.848 |
| asr_timings | 35.929 |
| ast_timings | 32.124 |
| describe_scenes | 16.939 |
| summarize_scenes | 19.789 |
| synthesize_synopsis | 22.726 |
| make_embedding | 3.900 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.186 |
| branch_yolo_total | 11.656 |
| branch_audio_total | 82.910 |
