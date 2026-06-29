# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:29:29 UTC | gehzWEPLjcc_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.078 | 0.703 | 44.513 | 10.520 | 10.149 | 17.814 | 3.347 |

## 2026-06-26 05:29:29 UTC | gehzWEPLjcc_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gehzWEPLjcc_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.078` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.703 |
| save_clips | - |
| sample_frames | 1.212 |
| caption_frames | 36.593 |
| sample_fps | 2.201 |
| detect_object_yolo | 8.562 |
| audio_scan | 8.742 |
| asr_timings | 8.303 |
| ast_timings | 27.460 |
| describe_scenes | 10.520 |
| summarize_scenes | 10.149 |
| synthesize_synopsis | 17.814 |
| make_embedding | 3.347 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.811 |
| branch_yolo_total | 10.768 |
| branch_audio_total | 44.513 |
