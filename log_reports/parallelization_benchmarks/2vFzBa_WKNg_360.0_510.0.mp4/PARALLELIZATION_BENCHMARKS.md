# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:00:06 UTC | 2vFzBa_WKNg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.870 | 3.755 | 62.120 | 10.689 | 9.821 | 7.990 | 4.802 |
| 2026-06-21 21:39:31 UTC | 2vFzBa_WKNg_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.023 | 3.824 | 62.995 | 9.069 | 10.059 | 6.507 | 4.707 |

## 2026-06-21 10:00:06 UTC | 2vFzBa_WKNg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.870` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.755 |
| save_clips | - |
| sample_frames | 6.682 |
| caption_frames | 48.291 |
| sample_fps | 12.444 |
| detect_object_yolo | 9.956 |
| audio_scan | 10.775 |
| asr_timings | 13.531 |
| ast_timings | 37.806 |
| describe_scenes | 10.689 |
| summarize_scenes | 9.821 |
| synthesize_synopsis | 7.990 |
| make_embedding | 4.802 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.979 |
| branch_yolo_total | 22.406 |
| branch_audio_total | 62.120 |

## 2026-06-21 21:39:31 UTC | 2vFzBa_WKNg_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.023` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.824 |
| save_clips | - |
| sample_frames | 6.624 |
| caption_frames | 48.902 |
| sample_fps | 12.561 |
| detect_object_yolo | 10.386 |
| audio_scan | 10.923 |
| asr_timings | 13.680 |
| ast_timings | 38.384 |
| describe_scenes | 9.069 |
| summarize_scenes | 10.059 |
| synthesize_synopsis | 6.507 |
| make_embedding | 4.707 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.532 |
| branch_yolo_total | 22.952 |
| branch_audio_total | 62.995 |
