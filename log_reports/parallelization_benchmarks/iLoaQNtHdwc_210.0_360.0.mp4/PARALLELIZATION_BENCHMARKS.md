# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:39:22 UTC | iLoaQNtHdwc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.855 | 0.702 | 48.096 | 15.630 | 10.997 | 16.258 | 2.826 |

## 2026-06-26 08:39:22 UTC | iLoaQNtHdwc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iLoaQNtHdwc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.855` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.702 |
| save_clips | - |
| sample_frames | 0.832 |
| caption_frames | 29.314 |
| sample_fps | 2.058 |
| detect_object_yolo | 7.651 |
| audio_scan | 15.302 |
| asr_timings | 11.127 |
| ast_timings | 21.658 |
| describe_scenes | 15.630 |
| summarize_scenes | 10.997 |
| synthesize_synopsis | 16.258 |
| make_embedding | 2.826 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.151 |
| branch_yolo_total | 9.715 |
| branch_audio_total | 48.096 |
