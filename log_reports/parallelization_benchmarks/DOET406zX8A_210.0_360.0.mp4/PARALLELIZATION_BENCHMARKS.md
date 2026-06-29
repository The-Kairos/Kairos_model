# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:35:03 UTC | DOET406zX8A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 80.329 | 0.633 | 32.928 | 7.848 | 6.177 | 7.641 | 3.064 |

## 2026-06-24 22:35:03 UTC | DOET406zX8A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DOET406zX8A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `80.329` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.633 |
| save_clips | - |
| sample_frames | 0.716 |
| caption_frames | 32.205 |
| sample_fps | 1.968 |
| detect_object_yolo | 7.514 |
| audio_scan | 3.863 |
| asr_timings | 0.000 |
| ast_timings | 7.272 |
| describe_scenes | 7.848 |
| summarize_scenes | 6.177 |
| synthesize_synopsis | 7.641 |
| make_embedding | 3.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.928 |
| branch_yolo_total | 9.488 |
| branch_audio_total | 11.144 |
