# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:30:04 UTC | zvEaxc9WD14_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.911 | 0.781 | 66.446 | 15.798 | 13.302 | 9.042 | 5.380 |

## 2026-06-27 06:30:04 UTC | zvEaxc9WD14_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zvEaxc9WD14_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.911` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.508 |
| caption_frames | 56.904 |
| sample_fps | 2.624 |
| detect_object_yolo | 11.665 |
| audio_scan | 12.961 |
| asr_timings | 9.983 |
| ast_timings | 43.493 |
| describe_scenes | 15.798 |
| summarize_scenes | 13.302 |
| synthesize_synopsis | 9.042 |
| make_embedding | 5.380 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.418 |
| branch_yolo_total | 14.295 |
| branch_audio_total | 66.446 |
