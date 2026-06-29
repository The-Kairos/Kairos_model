# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 16:09:12 UTC | ljcGQQxxO1E_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 1439.224 | 0.633 | 1346.913 | 16.471 | 5.291 | 19.270 | 3.425 |

## 2026-06-26 16:09:12 UTC | ljcGQQxxO1E_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ljcGQQxxO1E_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1439.224` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.633 |
| save_clips | - |
| sample_frames | 0.928 |
| caption_frames | 34.918 |
| sample_fps | 2.007 |
| detect_object_yolo | 7.911 |
| audio_scan | 7.654 |
| asr_timings | 1311.462 |
| ast_timings | 27.789 |
| describe_scenes | 16.471 |
| summarize_scenes | 5.291 |
| synthesize_synopsis | 19.270 |
| make_embedding | 3.425 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.853 |
| branch_yolo_total | 9.923 |
| branch_audio_total | 1346.913 |
