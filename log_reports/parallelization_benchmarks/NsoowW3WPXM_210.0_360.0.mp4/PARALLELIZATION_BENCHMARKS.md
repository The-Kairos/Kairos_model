# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:03:00 UTC | NsoowW3WPXM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 218.003 | 0.862 | 64.981 | 34.859 | 12.365 | 21.993 | 5.442 |

## 2026-06-25 11:03:00 UTC | NsoowW3WPXM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/NsoowW3WPXM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `218.003` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.862 |
| save_clips | - |
| sample_frames | 1.606 |
| caption_frames | 60.351 |
| sample_fps | 2.635 |
| detect_object_yolo | 11.459 |
| audio_scan | 13.087 |
| asr_timings | 7.601 |
| ast_timings | 44.285 |
| describe_scenes | 34.859 |
| summarize_scenes | 12.365 |
| synthesize_synopsis | 21.993 |
| make_embedding | 5.442 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.963 |
| branch_yolo_total | 14.101 |
| branch_audio_total | 64.981 |
