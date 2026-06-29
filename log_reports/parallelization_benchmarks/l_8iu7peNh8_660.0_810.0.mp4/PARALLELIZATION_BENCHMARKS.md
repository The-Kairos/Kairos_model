# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:33:06 UTC | l_8iu7peNh8_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 201.672 | 0.697 | 88.011 | 20.994 | 17.485 | 20.833 | 3.324 |

## 2026-06-26 15:33:06 UTC | l_8iu7peNh8_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l_8iu7peNh8_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `201.672` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.697 |
| save_clips | - |
| sample_frames | 1.363 |
| caption_frames | 36.828 |
| sample_fps | 2.236 |
| detect_object_yolo | 8.489 |
| audio_scan | 12.978 |
| asr_timings | 47.270 |
| ast_timings | 27.755 |
| describe_scenes | 20.994 |
| summarize_scenes | 17.485 |
| synthesize_synopsis | 20.833 |
| make_embedding | 3.324 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.197 |
| branch_yolo_total | 10.731 |
| branch_audio_total | 88.011 |
