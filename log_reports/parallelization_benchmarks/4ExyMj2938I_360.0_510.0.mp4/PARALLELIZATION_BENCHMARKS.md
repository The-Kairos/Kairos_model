# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:52:03 UTC | 4ExyMj2938I_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.102 | 0.796 | 52.929 | 31.928 | 28.090 | 14.982 | 3.619 |
| 2026-06-24 10:46:39 UTC | 4ExyMj2938I_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.585 | 0.764 | 52.798 | 22.856 | 13.232 | 15.208 | 3.583 |

## 2026-06-23 16:52:03 UTC | 4ExyMj2938I_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4ExyMj2938I_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.102` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.008 |
| caption_frames | 37.109 |
| sample_fps | 2.221 |
| detect_object_yolo | 9.049 |
| audio_scan | 14.784 |
| asr_timings | 9.026 |
| ast_timings | 29.110 |
| describe_scenes | 31.928 |
| summarize_scenes | 28.090 |
| synthesize_synopsis | 14.982 |
| make_embedding | 3.619 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.123 |
| branch_yolo_total | 11.276 |
| branch_audio_total | 52.929 |

## 2026-06-24 10:46:39 UTC | 4ExyMj2938I_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4ExyMj2938I_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.585` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.764 |
| save_clips | - |
| sample_frames | 1.025 |
| caption_frames | 37.358 |
| sample_fps | 2.242 |
| detect_object_yolo | 9.129 |
| audio_scan | 14.963 |
| asr_timings | 8.585 |
| ast_timings | 29.240 |
| describe_scenes | 22.856 |
| summarize_scenes | 13.232 |
| synthesize_synopsis | 15.208 |
| make_embedding | 3.583 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.389 |
| branch_yolo_total | 11.377 |
| branch_audio_total | 52.798 |
