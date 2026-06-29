# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:27:51 UTC | IG7PxoN3Zao_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.574 | 0.660 | 58.443 | 15.913 | 10.135 | 9.556 | 5.052 |

## 2026-06-25 04:27:51 UTC | IG7PxoN3Zao_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IG7PxoN3Zao_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.574` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.660 |
| save_clips | - |
| sample_frames | 1.341 |
| caption_frames | 54.787 |
| sample_fps | 2.343 |
| detect_object_yolo | 10.915 |
| audio_scan | 6.455 |
| asr_timings | 10.477 |
| ast_timings | 41.503 |
| describe_scenes | 15.913 |
| summarize_scenes | 10.135 |
| synthesize_synopsis | 9.556 |
| make_embedding | 5.052 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.135 |
| branch_yolo_total | 13.264 |
| branch_audio_total | 58.443 |
