# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:49:58 UTC | iLoaQNtHdwc_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.232 | 0.706 | 51.895 | 15.191 | 17.368 | 20.256 | 2.538 |

## 2026-06-26 08:49:58 UTC | iLoaQNtHdwc_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iLoaQNtHdwc_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.232` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.706 |
| save_clips | - |
| sample_frames | 0.689 |
| caption_frames | 26.614 |
| sample_fps | 1.905 |
| detect_object_yolo | 6.659 |
| audio_scan | 11.832 |
| asr_timings | 21.439 |
| ast_timings | 18.614 |
| describe_scenes | 15.191 |
| summarize_scenes | 17.368 |
| synthesize_synopsis | 20.256 |
| make_embedding | 2.538 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.308 |
| branch_yolo_total | 8.570 |
| branch_audio_total | 51.895 |
