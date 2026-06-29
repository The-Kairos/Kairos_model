# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:14:48 UTC | cIElQi5Sfos_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.294 | 0.793 | 48.325 | 11.723 | 15.459 | 12.596 | 3.280 |

## 2026-06-26 02:14:48 UTC | cIElQi5Sfos_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/cIElQi5Sfos_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.294` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.092 |
| caption_frames | 37.805 |
| sample_fps | 2.239 |
| detect_object_yolo | 8.579 |
| audio_scan | 9.762 |
| asr_timings | 11.079 |
| ast_timings | 27.477 |
| describe_scenes | 11.723 |
| summarize_scenes | 15.459 |
| synthesize_synopsis | 12.596 |
| make_embedding | 3.280 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.903 |
| branch_yolo_total | 10.824 |
| branch_audio_total | 48.325 |
