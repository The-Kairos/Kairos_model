# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:38:38 UTC | ooJUCHRH0OM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.888 | 0.805 | 60.745 | 10.688 | 23.061 | 10.006 | 4.677 |

## 2026-06-28 07:38:38 UTC | ooJUCHRH0OM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ooJUCHRH0OM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.888` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.613 |
| caption_frames | 52.474 |
| sample_fps | 2.503 |
| detect_object_yolo | 10.852 |
| audio_scan | 12.862 |
| asr_timings | 9.519 |
| ast_timings | 38.355 |
| describe_scenes | 10.688 |
| summarize_scenes | 23.061 |
| synthesize_synopsis | 10.006 |
| make_embedding | 4.677 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.094 |
| branch_yolo_total | 13.361 |
| branch_audio_total | 60.745 |
