# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:53:10 UTC | i9scCMPwu8I_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 197.114 | 0.714 | 55.501 | 19.275 | 38.606 | 29.677 | 3.371 |

## 2026-06-26 07:53:10 UTC | i9scCMPwu8I_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i9scCMPwu8I_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `197.114` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.714 |
| save_clips | - |
| sample_frames | 1.312 |
| caption_frames | 35.966 |
| sample_fps | 2.182 |
| detect_object_yolo | 9.038 |
| audio_scan | 16.358 |
| asr_timings | 11.718 |
| ast_timings | 27.417 |
| describe_scenes | 19.275 |
| summarize_scenes | 38.606 |
| synthesize_synopsis | 29.677 |
| make_embedding | 3.371 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.283 |
| branch_yolo_total | 11.226 |
| branch_audio_total | 55.501 |
