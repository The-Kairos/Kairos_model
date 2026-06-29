# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:10:59 UTC | hpBZbgdZJEo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 286.801 | 0.803 | 85.805 | 38.143 | 42.847 | 35.663 | 5.709 |

## 2026-06-26 07:10:59 UTC | hpBZbgdZJEo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hpBZbgdZJEo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `286.801` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.685 |
| caption_frames | 60.642 |
| sample_fps | 2.532 |
| detect_object_yolo | 11.556 |
| audio_scan | 11.863 |
| asr_timings | 27.351 |
| ast_timings | 46.582 |
| describe_scenes | 38.143 |
| summarize_scenes | 42.847 |
| synthesize_synopsis | 35.663 |
| make_embedding | 5.709 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.333 |
| branch_yolo_total | 14.093 |
| branch_audio_total | 85.805 |
