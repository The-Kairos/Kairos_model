# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:02:55 UTC | QP2e5M7SZy4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.675 | 0.801 | 56.013 | 17.967 | 11.714 | 23.088 | 3.916 |

## 2026-06-25 15:02:55 UTC | QP2e5M7SZy4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QP2e5M7SZy4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.675` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.025 |
| caption_frames | 43.391 |
| sample_fps | 2.312 |
| detect_object_yolo | 9.031 |
| audio_scan | 12.242 |
| asr_timings | 11.436 |
| ast_timings | 32.327 |
| describe_scenes | 17.967 |
| summarize_scenes | 11.714 |
| synthesize_synopsis | 23.088 |
| make_embedding | 3.916 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.423 |
| branch_yolo_total | 11.349 |
| branch_audio_total | 56.013 |
