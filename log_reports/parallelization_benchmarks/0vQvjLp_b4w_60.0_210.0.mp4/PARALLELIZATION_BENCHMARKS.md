# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 12:36:31 UTC | 0vQvjLp_b4w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 239.563 | 0.750 | 41.723 | 10.870 | 9.470 | 140.068 | 2.576 |
| 2026-06-27 14:23:49 UTC | 0vQvjLp_b4w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 101.801 | 0.774 | 42.747 | 5.135 | 5.274 | 10.159 | 2.585 |

## 2026-06-23 12:36:31 UTC | 0vQvjLp_b4w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0vQvjLp_b4w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `239.563` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.750 |
| save_clips | - |
| sample_frames | 0.565 |
| caption_frames | 23.364 |
| sample_fps | 1.977 |
| detect_object_yolo | 6.825 |
| audio_scan | 12.664 |
| asr_timings | 10.891 |
| ast_timings | 18.159 |
| describe_scenes | 10.870 |
| summarize_scenes | 9.470 |
| synthesize_synopsis | 140.068 |
| make_embedding | 2.576 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.934 |
| branch_yolo_total | 8.808 |
| branch_audio_total | 41.723 |

## 2026-06-27 14:23:49 UTC | 0vQvjLp_b4w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0vQvjLp_b4w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `101.801` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 0.571 |
| caption_frames | 24.235 |
| sample_fps | 1.994 |
| detect_object_yolo | 6.929 |
| audio_scan | 12.858 |
| asr_timings | 11.436 |
| ast_timings | 18.444 |
| describe_scenes | 5.135 |
| summarize_scenes | 5.274 |
| synthesize_synopsis | 10.159 |
| make_embedding | 2.585 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.812 |
| branch_yolo_total | 8.928 |
| branch_audio_total | 42.747 |
