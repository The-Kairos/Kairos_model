# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:39:16 UTC | GKyQFpBxCK4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.391 | 0.658 | 49.778 | 11.598 | 8.239 | 14.813 | 3.237 |

## 2026-06-25 01:39:16 UTC | GKyQFpBxCK4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GKyQFpBxCK4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.391` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.658 |
| save_clips | - |
| sample_frames | 0.961 |
| caption_frames | 37.866 |
| sample_fps | 2.085 |
| detect_object_yolo | 8.728 |
| audio_scan | 12.823 |
| asr_timings | 10.137 |
| ast_timings | 26.810 |
| describe_scenes | 11.598 |
| summarize_scenes | 8.239 |
| synthesize_synopsis | 14.813 |
| make_embedding | 3.237 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.834 |
| branch_yolo_total | 10.820 |
| branch_audio_total | 49.778 |
