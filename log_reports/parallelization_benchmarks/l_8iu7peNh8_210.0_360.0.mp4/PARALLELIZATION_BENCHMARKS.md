# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:19:46 UTC | l_8iu7peNh8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.671 | 0.673 | 94.881 | 13.715 | 13.515 | 17.851 | 2.562 |

## 2026-06-26 15:19:46 UTC | l_8iu7peNh8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l_8iu7peNh8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.671` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.673 |
| save_clips | - |
| sample_frames | 0.662 |
| caption_frames | 24.476 |
| sample_fps | 1.916 |
| detect_object_yolo | 6.935 |
| audio_scan | 9.875 |
| asr_timings | 65.948 |
| ast_timings | 19.049 |
| describe_scenes | 13.715 |
| summarize_scenes | 13.515 |
| synthesize_synopsis | 17.851 |
| make_embedding | 2.562 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.144 |
| branch_yolo_total | 8.857 |
| branch_audio_total | 94.881 |
