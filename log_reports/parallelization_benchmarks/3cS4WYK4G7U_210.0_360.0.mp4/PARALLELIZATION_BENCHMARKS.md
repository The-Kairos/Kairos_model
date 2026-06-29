# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:22:56 UTC | 3cS4WYK4G7U_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 99.858 | 0.766 | 29.008 | 9.157 | 8.831 | 33.814 | 1.337 |
| 2026-06-24 10:18:18 UTC | 3cS4WYK4G7U_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 85.505 | 0.793 | 28.564 | 8.878 | 7.753 | 21.140 | 1.312 |

## 2026-06-23 16:22:56 UTC | 3cS4WYK4G7U_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `99.858` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 0.088 |
| caption_frames | 8.127 |
| sample_fps | 1.737 |
| detect_object_yolo | 5.616 |
| audio_scan | 14.726 |
| asr_timings | 9.948 |
| ast_timings | 4.326 |
| describe_scenes | 9.157 |
| summarize_scenes | 8.831 |
| synthesize_synopsis | 33.814 |
| make_embedding | 1.337 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.220 |
| branch_yolo_total | 7.359 |
| branch_audio_total | 29.008 |

## 2026-06-24 10:18:18 UTC | 3cS4WYK4G7U_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `85.505` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 0.092 |
| caption_frames | 8.268 |
| sample_fps | 1.767 |
| detect_object_yolo | 5.548 |
| audio_scan | 14.764 |
| asr_timings | 9.477 |
| ast_timings | 4.315 |
| describe_scenes | 8.878 |
| summarize_scenes | 7.753 |
| synthesize_synopsis | 21.140 |
| make_embedding | 1.312 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.366 |
| branch_yolo_total | 7.321 |
| branch_audio_total | 28.564 |
