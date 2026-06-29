# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:16:15 UTC | OkHhVRpCOxA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 274.293 | 0.828 | 97.084 | 30.106 | 30.883 | 26.313 | 6.444 |

## 2026-06-25 12:16:15 UTC | OkHhVRpCOxA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OkHhVRpCOxA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `274.293` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.828 |
| save_clips | - |
| sample_frames | 1.876 |
| caption_frames | 64.396 |
| sample_fps | 2.729 |
| detect_object_yolo | 12.218 |
| audio_scan | 13.310 |
| asr_timings | 33.899 |
| ast_timings | 49.866 |
| describe_scenes | 30.106 |
| summarize_scenes | 30.883 |
| synthesize_synopsis | 26.313 |
| make_embedding | 6.444 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.278 |
| branch_yolo_total | 14.953 |
| branch_audio_total | 97.084 |
