# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:48:43 UTC | uGymr9TVaGI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 78.296 | 0.861 | 33.032 | 3.767 | 3.066 | 12.112 | 1.834 |

## 2026-06-27 00:48:43 UTC | uGymr9TVaGI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uGymr9TVaGI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `78.296` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.861 |
| save_clips | - |
| sample_frames | 0.282 |
| caption_frames | 13.980 |
| sample_fps | 1.902 |
| detect_object_yolo | 5.964 |
| audio_scan | 14.981 |
| asr_timings | 6.914 |
| ast_timings | 11.129 |
| describe_scenes | 3.767 |
| summarize_scenes | 3.066 |
| synthesize_synopsis | 12.112 |
| make_embedding | 1.834 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.268 |
| branch_yolo_total | 7.872 |
| branch_audio_total | 33.032 |
