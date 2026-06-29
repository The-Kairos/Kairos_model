# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 17:21:17 UTC | 4jNdZg348kM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 214.722 | 0.796 | 92.566 | 27.353 | 14.008 | 12.562 | 4.495 |
| 2026-06-24 11:14:01 UTC | 4jNdZg348kM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.203 | 0.793 | 92.857 | 19.299 | 18.663 | 14.866 | 4.490 |

## 2026-06-23 17:21:17 UTC | 4jNdZg348kM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4jNdZg348kM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `214.722` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.541 |
| caption_frames | 47.585 |
| sample_fps | 2.441 |
| detect_object_yolo | 9.965 |
| audio_scan | 15.752 |
| asr_timings | 38.206 |
| ast_timings | 38.599 |
| describe_scenes | 27.353 |
| summarize_scenes | 14.008 |
| synthesize_synopsis | 12.562 |
| make_embedding | 4.495 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.132 |
| branch_yolo_total | 12.412 |
| branch_audio_total | 92.566 |

## 2026-06-24 11:14:01 UTC | 4jNdZg348kM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4jNdZg348kM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.203` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.539 |
| caption_frames | 48.917 |
| sample_fps | 2.435 |
| detect_object_yolo | 9.924 |
| audio_scan | 15.770 |
| asr_timings | 38.788 |
| ast_timings | 38.289 |
| describe_scenes | 19.299 |
| summarize_scenes | 18.663 |
| synthesize_synopsis | 14.866 |
| make_embedding | 4.490 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.462 |
| branch_yolo_total | 12.365 |
| branch_audio_total | 92.857 |
