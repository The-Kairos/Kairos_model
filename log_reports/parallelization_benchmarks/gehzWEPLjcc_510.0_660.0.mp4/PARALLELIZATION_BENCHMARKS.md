# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:23:58 UTC | gehzWEPLjcc_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 149.663 | 0.696 | 50.527 | 11.587 | 8.688 | 15.645 | 3.846 |

## 2026-06-26 05:23:58 UTC | gehzWEPLjcc_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gehzWEPLjcc_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `149.663` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.696 |
| save_clips | - |
| sample_frames | 1.435 |
| caption_frames | 43.978 |
| sample_fps | 2.324 |
| detect_object_yolo | 9.437 |
| audio_scan | 8.826 |
| asr_timings | 8.362 |
| ast_timings | 33.330 |
| describe_scenes | 11.587 |
| summarize_scenes | 8.688 |
| synthesize_synopsis | 15.645 |
| make_embedding | 3.846 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.420 |
| branch_yolo_total | 11.768 |
| branch_audio_total | 50.527 |
