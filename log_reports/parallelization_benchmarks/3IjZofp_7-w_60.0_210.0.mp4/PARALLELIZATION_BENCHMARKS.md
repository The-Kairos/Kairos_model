# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:01:43 UTC | 3IjZofp_7-w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.701 | 0.746 | 37.798 | 11.276 | 20.823 | 33.967 | 1.793 |
| 2026-06-24 09:58:41 UTC | 3IjZofp_7-w_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.340 | 0.782 | 37.319 | 11.853 | 13.748 | 21.266 | 1.839 |

## 2026-06-23 16:01:43 UTC | 3IjZofp_7-w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3IjZofp_7-w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.701` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.746 |
| save_clips | - |
| sample_frames | 0.259 |
| caption_frames | 15.101 |
| sample_fps | 1.826 |
| detect_object_yolo | 6.729 |
| audio_scan | 14.751 |
| asr_timings | 12.989 |
| ast_timings | 10.050 |
| describe_scenes | 11.276 |
| summarize_scenes | 20.823 |
| synthesize_synopsis | 33.967 |
| make_embedding | 1.793 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.365 |
| branch_yolo_total | 8.560 |
| branch_audio_total | 37.798 |

## 2026-06-24 09:58:41 UTC | 3IjZofp_7-w_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3IjZofp_7-w_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.340` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.256 |
| caption_frames | 15.148 |
| sample_fps | 1.855 |
| detect_object_yolo | 6.871 |
| audio_scan | 14.992 |
| asr_timings | 12.261 |
| ast_timings | 10.058 |
| describe_scenes | 11.853 |
| summarize_scenes | 13.748 |
| synthesize_synopsis | 21.266 |
| make_embedding | 1.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.410 |
| branch_yolo_total | 8.732 |
| branch_audio_total | 37.319 |
