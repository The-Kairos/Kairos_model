# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:29:35 UTC | HQBXQyT8UoI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.014 | 0.816 | 68.326 | 13.221 | 16.127 | 19.041 | 5.336 |

## 2026-06-25 03:29:35 UTC | HQBXQyT8UoI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HQBXQyT8UoI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.014` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 1.527 |
| caption_frames | 55.533 |
| sample_fps | 2.532 |
| detect_object_yolo | 11.137 |
| audio_scan | 15.941 |
| asr_timings | 9.018 |
| ast_timings | 43.359 |
| describe_scenes | 13.221 |
| summarize_scenes | 16.127 |
| synthesize_synopsis | 19.041 |
| make_embedding | 5.336 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.066 |
| branch_yolo_total | 13.675 |
| branch_audio_total | 68.326 |
