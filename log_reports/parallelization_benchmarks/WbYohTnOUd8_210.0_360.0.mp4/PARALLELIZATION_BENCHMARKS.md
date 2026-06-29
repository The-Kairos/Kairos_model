# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:43:39 UTC | WbYohTnOUd8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 218.999 | 0.698 | 93.040 | 19.932 | 19.591 | 9.633 | 5.042 |

## 2026-06-25 20:43:39 UTC | WbYohTnOUd8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WbYohTnOUd8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `218.999` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.698 |
| save_clips | - |
| sample_frames | 1.647 |
| caption_frames | 54.834 |
| sample_fps | 2.300 |
| detect_object_yolo | 10.880 |
| audio_scan | 13.847 |
| asr_timings | 37.937 |
| ast_timings | 41.248 |
| describe_scenes | 19.932 |
| summarize_scenes | 19.591 |
| synthesize_synopsis | 9.633 |
| make_embedding | 5.042 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.486 |
| branch_yolo_total | 13.186 |
| branch_audio_total | 93.040 |
