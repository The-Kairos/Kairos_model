# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:20:01 UTC | COXt_GfXa2M_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 118.548 | 0.665 | 41.107 | 11.192 | 7.248 | 14.665 | 2.785 |

## 2026-06-24 20:20:01 UTC | COXt_GfXa2M_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/COXt_GfXa2M_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `118.548` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 0.813 |
| caption_frames | 29.109 |
| sample_fps | 1.930 |
| detect_object_yolo | 7.601 |
| audio_scan | 9.715 |
| asr_timings | 9.946 |
| ast_timings | 21.437 |
| describe_scenes | 11.192 |
| summarize_scenes | 7.248 |
| synthesize_synopsis | 14.665 |
| make_embedding | 2.785 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.927 |
| branch_yolo_total | 9.537 |
| branch_audio_total | 41.107 |
