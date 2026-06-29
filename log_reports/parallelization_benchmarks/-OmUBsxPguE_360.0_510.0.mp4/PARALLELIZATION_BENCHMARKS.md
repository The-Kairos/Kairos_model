# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:08:57 UTC | -OmUBsxPguE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.993 | 0.656 | 43.688 | 16.700 | 10.279 | 30.301 | 2.777 |

## 2026-06-24 08:08:57 UTC | -OmUBsxPguE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-OmUBsxPguE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.993` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 0.753 |
| caption_frames | 28.188 |
| sample_fps | 1.973 |
| detect_object_yolo | 7.338 |
| audio_scan | 12.702 |
| asr_timings | 10.289 |
| ast_timings | 20.687 |
| describe_scenes | 16.700 |
| summarize_scenes | 10.279 |
| synthesize_synopsis | 30.301 |
| make_embedding | 2.777 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.946 |
| branch_yolo_total | 9.317 |
| branch_audio_total | 43.688 |
