# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:42:01 UTC | uCZAfLBvPVo_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.132 | 0.653 | 59.296 | 5.202 | 5.031 | 7.804 | 2.016 |

## 2026-06-27 00:42:01 UTC | uCZAfLBvPVo_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uCZAfLBvPVo_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.132` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.653 |
| save_clips | - |
| sample_frames | 0.485 |
| caption_frames | 16.610 |
| sample_fps | 1.812 |
| detect_object_yolo | 6.813 |
| audio_scan | 13.828 |
| asr_timings | 32.504 |
| ast_timings | 12.955 |
| describe_scenes | 5.202 |
| summarize_scenes | 5.031 |
| synthesize_synopsis | 7.804 |
| make_embedding | 2.016 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.101 |
| branch_yolo_total | 8.630 |
| branch_audio_total | 59.296 |
