# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:22:36 UTC | l_8iu7peNh8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.053 | 0.685 | 66.664 | 20.531 | 8.924 | 22.427 | 3.059 |

## 2026-06-26 15:22:36 UTC | l_8iu7peNh8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l_8iu7peNh8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.053` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.685 |
| save_clips | - |
| sample_frames | 1.054 |
| caption_frames | 34.287 |
| sample_fps | 2.056 |
| detect_object_yolo | 7.925 |
| audio_scan | 12.975 |
| asr_timings | 28.640 |
| ast_timings | 25.040 |
| describe_scenes | 20.531 |
| summarize_scenes | 8.924 |
| synthesize_synopsis | 22.427 |
| make_embedding | 3.059 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.347 |
| branch_yolo_total | 9.987 |
| branch_audio_total | 66.664 |
