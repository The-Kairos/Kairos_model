# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:26:58 UTC | zvEaxc9WD14_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.281 | 0.796 | 62.118 | 14.368 | 11.098 | 6.028 | 5.119 |

## 2026-06-27 06:26:58 UTC | zvEaxc9WD14_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zvEaxc9WD14_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.281` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.434 |
| caption_frames | 55.533 |
| sample_fps | 2.516 |
| detect_object_yolo | 10.843 |
| audio_scan | 10.675 |
| asr_timings | 10.291 |
| ast_timings | 41.143 |
| describe_scenes | 14.368 |
| summarize_scenes | 11.098 |
| synthesize_synopsis | 6.028 |
| make_embedding | 5.119 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.973 |
| branch_yolo_total | 13.365 |
| branch_audio_total | 62.118 |
