# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 19:38:06 UTC | VFhXXjTwkKw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.681 | 0.814 | 42.093 | 13.663 | 8.584 | 24.420 | 2.784 |

## 2026-06-25 19:38:06 UTC | VFhXXjTwkKw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VFhXXjTwkKw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.681` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 0.653 |
| caption_frames | 27.489 |
| sample_fps | 2.020 |
| detect_object_yolo | 7.761 |
| audio_scan | 11.772 |
| asr_timings | 8.555 |
| ast_timings | 21.757 |
| describe_scenes | 13.663 |
| summarize_scenes | 8.584 |
| synthesize_synopsis | 24.420 |
| make_embedding | 2.784 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.148 |
| branch_yolo_total | 9.787 |
| branch_audio_total | 42.093 |
