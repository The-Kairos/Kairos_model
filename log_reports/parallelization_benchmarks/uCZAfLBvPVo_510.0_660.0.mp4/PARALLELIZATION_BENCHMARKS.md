# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:37:28 UTC | uCZAfLBvPVo_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.322 | 0.657 | 68.158 | 11.830 | 18.594 | 9.955 | 3.278 |

## 2026-06-27 00:37:28 UTC | uCZAfLBvPVo_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uCZAfLBvPVo_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.322` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 1.269 |
| caption_frames | 38.079 |
| sample_fps | 2.196 |
| detect_object_yolo | 8.858 |
| audio_scan | 7.576 |
| asr_timings | 32.478 |
| ast_timings | 28.096 |
| describe_scenes | 11.830 |
| summarize_scenes | 18.594 |
| synthesize_synopsis | 9.955 |
| make_embedding | 3.278 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.353 |
| branch_yolo_total | 11.059 |
| branch_audio_total | 68.158 |
