# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:35:04 UTC | N4VtpYgZLVg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 209.095 | 0.674 | 61.524 | 26.597 | 29.620 | 11.144 | 5.218 |

## 2026-06-25 10:35:04 UTC | N4VtpYgZLVg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/N4VtpYgZLVg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `209.095` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 1.793 |
| caption_frames | 57.124 |
| sample_fps | 2.425 |
| detect_object_yolo | 11.517 |
| audio_scan | 11.790 |
| asr_timings | 8.383 |
| ast_timings | 41.343 |
| describe_scenes | 26.597 |
| summarize_scenes | 29.620 |
| synthesize_synopsis | 11.144 |
| make_embedding | 5.218 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.923 |
| branch_yolo_total | 13.948 |
| branch_audio_total | 61.524 |
