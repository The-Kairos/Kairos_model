# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:20:31 UTC | oc6imtgs6ig_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.158 | 0.775 | 55.530 | 8.093 | 16.230 | 10.496 | 3.834 |

## 2026-06-28 07:20:31 UTC | oc6imtgs6ig_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/oc6imtgs6ig_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.158` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 1.135 |
| caption_frames | 38.110 |
| sample_fps | 2.272 |
| detect_object_yolo | 9.237 |
| audio_scan | 13.789 |
| asr_timings | 9.486 |
| ast_timings | 32.246 |
| describe_scenes | 8.093 |
| summarize_scenes | 16.230 |
| synthesize_synopsis | 10.496 |
| make_embedding | 3.834 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.251 |
| branch_yolo_total | 11.516 |
| branch_audio_total | 55.530 |
