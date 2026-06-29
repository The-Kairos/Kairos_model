# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:53:43 UTC | uGymr9TVaGI_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 65.448 | 0.761 | 28.518 | 3.634 | 7.633 | 6.613 | 1.228 |

## 2026-06-27 00:53:43 UTC | uGymr9TVaGI_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uGymr9TVaGI_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `65.448` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.761 |
| save_clips | - |
| sample_frames | 0.127 |
| caption_frames | 8.033 |
| sample_fps | 1.841 |
| detect_object_yolo | 5.645 |
| audio_scan | 12.857 |
| asr_timings | 10.283 |
| ast_timings | 5.369 |
| describe_scenes | 3.634 |
| summarize_scenes | 7.633 |
| synthesize_synopsis | 6.613 |
| make_embedding | 1.228 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.166 |
| branch_yolo_total | 7.492 |
| branch_audio_total | 28.518 |
