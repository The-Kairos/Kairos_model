# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:03:57 UTC | fBFSes_K4u0_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 89.411 | 0.786 | 52.673 | 3.619 | 2.292 | 9.706 | 1.545 |

## 2026-06-26 04:03:57 UTC | fBFSes_K4u0_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fBFSes_K4u0_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `89.411` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.138 |
| caption_frames | 9.562 |
| sample_fps | 1.870 |
| detect_object_yolo | 5.841 |
| audio_scan | 16.260 |
| asr_timings | 29.294 |
| ast_timings | 7.111 |
| describe_scenes | 3.619 |
| summarize_scenes | 2.292 |
| synthesize_synopsis | 9.706 |
| make_embedding | 1.545 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 9.706 |
| branch_yolo_total | 7.717 |
| branch_audio_total | 52.673 |
