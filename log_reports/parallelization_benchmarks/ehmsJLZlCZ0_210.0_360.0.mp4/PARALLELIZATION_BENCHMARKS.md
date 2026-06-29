# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:51:00 UTC | ehmsJLZlCZ0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 174.705 | 0.666 | 91.072 | 11.071 | 10.390 | 19.346 | 2.594 |

## 2026-06-26 03:51:00 UTC | ehmsJLZlCZ0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ehmsJLZlCZ0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `174.705` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.666 |
| save_clips | - |
| sample_frames | 0.722 |
| caption_frames | 27.409 |
| sample_fps | 1.962 |
| detect_object_yolo | 8.031 |
| audio_scan | 7.693 |
| asr_timings | 64.410 |
| ast_timings | 18.959 |
| describe_scenes | 11.071 |
| summarize_scenes | 10.390 |
| synthesize_synopsis | 19.346 |
| make_embedding | 2.594 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.136 |
| branch_yolo_total | 9.999 |
| branch_audio_total | 91.072 |
