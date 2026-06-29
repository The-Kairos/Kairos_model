# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:30:46 UTC | 75Tp9uriob8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.058 | 0.664 | 44.057 | 22.108 | 13.565 | 29.062 | 3.109 |

## 2026-06-24 13:30:46 UTC | 75Tp9uriob8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/75Tp9uriob8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.058` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 0.798 |
| caption_frames | 30.637 |
| sample_fps | 1.983 |
| detect_object_yolo | 7.674 |
| audio_scan | 12.740 |
| asr_timings | 7.547 |
| ast_timings | 23.762 |
| describe_scenes | 22.108 |
| summarize_scenes | 13.565 |
| synthesize_synopsis | 29.062 |
| make_embedding | 3.109 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.441 |
| branch_yolo_total | 9.662 |
| branch_audio_total | 44.057 |
