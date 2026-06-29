# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:36:42 UTC | IG7PxoN3Zao_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.813 | 0.684 | 57.155 | 19.287 | 8.361 | 9.643 | 4.958 |

## 2026-06-25 04:36:42 UTC | IG7PxoN3Zao_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IG7PxoN3Zao_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.813` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 1.434 |
| caption_frames | 54.683 |
| sample_fps | 2.358 |
| detect_object_yolo | 10.823 |
| audio_scan | 6.453 |
| asr_timings | 10.156 |
| ast_timings | 40.537 |
| describe_scenes | 19.287 |
| summarize_scenes | 8.361 |
| synthesize_synopsis | 9.643 |
| make_embedding | 4.958 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.123 |
| branch_yolo_total | 13.187 |
| branch_audio_total | 57.155 |
