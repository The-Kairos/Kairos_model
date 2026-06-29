# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:59:31 UTC | 3IjZofp_7-w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.088 | 0.892 | 44.650 | 28.467 | 11.629 | 24.676 | 3.147 |
| 2026-06-24 09:56:48 UTC | 3IjZofp_7-w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 219.597 | 0.784 | 44.542 | 22.700 | 22.983 | 81.101 | 3.033 |

## 2026-06-23 15:59:31 UTC | 3IjZofp_7-w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3IjZofp_7-w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.088` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.892 |
| save_clips | - |
| sample_frames | 0.881 |
| caption_frames | 30.895 |
| sample_fps | 2.100 |
| detect_object_yolo | 8.381 |
| audio_scan | 11.693 |
| asr_timings | 9.601 |
| ast_timings | 23.348 |
| describe_scenes | 28.467 |
| summarize_scenes | 11.629 |
| synthesize_synopsis | 24.676 |
| make_embedding | 3.147 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.782 |
| branch_yolo_total | 10.486 |
| branch_audio_total | 44.650 |

## 2026-06-24 09:56:48 UTC | 3IjZofp_7-w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3IjZofp_7-w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `219.597` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.784 |
| save_clips | - |
| sample_frames | 0.911 |
| caption_frames | 31.220 |
| sample_fps | 2.151 |
| detect_object_yolo | 8.704 |
| audio_scan | 11.852 |
| asr_timings | 8.962 |
| ast_timings | 23.720 |
| describe_scenes | 22.700 |
| summarize_scenes | 22.983 |
| synthesize_synopsis | 81.101 |
| make_embedding | 3.033 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.137 |
| branch_yolo_total | 10.860 |
| branch_audio_total | 44.542 |
