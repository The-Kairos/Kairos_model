# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:07:40 UTC | NsoowW3WPXM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 278.597 | 0.854 | 71.866 | 36.232 | 47.075 | 37.387 | 5.741 |

## 2026-06-25 11:07:40 UTC | NsoowW3WPXM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/NsoowW3WPXM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `278.597` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.854 |
| save_clips | - |
| sample_frames | 1.797 |
| caption_frames | 61.523 |
| sample_fps | 2.731 |
| detect_object_yolo | 11.925 |
| audio_scan | 15.264 |
| asr_timings | 9.459 |
| ast_timings | 47.134 |
| describe_scenes | 36.232 |
| summarize_scenes | 47.075 |
| synthesize_synopsis | 37.387 |
| make_embedding | 5.741 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.326 |
| branch_yolo_total | 14.662 |
| branch_audio_total | 71.866 |
