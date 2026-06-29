# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:33:23 UTC | 3cS4WYK4G7U_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 216.708 | 0.769 | 63.580 | 28.982 | 15.401 | 37.575 | 4.688 |
| 2026-06-24 10:28:28 UTC | 3cS4WYK4G7U_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.225 | 0.819 | 63.167 | 20.821 | 24.192 | 18.326 | 4.790 |

## 2026-06-23 16:33:23 UTC | 3cS4WYK4G7U_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `216.708` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.769 |
| save_clips | - |
| sample_frames | 1.787 |
| caption_frames | 50.381 |
| sample_fps | 2.478 |
| detect_object_yolo | 9.694 |
| audio_scan | 14.788 |
| asr_timings | 11.548 |
| ast_timings | 37.236 |
| describe_scenes | 28.982 |
| summarize_scenes | 15.401 |
| synthesize_synopsis | 37.575 |
| make_embedding | 4.688 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.174 |
| branch_yolo_total | 12.177 |
| branch_audio_total | 63.580 |

## 2026-06-24 10:28:28 UTC | 3cS4WYK4G7U_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.225` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 1.805 |
| caption_frames | 51.619 |
| sample_fps | 2.505 |
| detect_object_yolo | 9.783 |
| audio_scan | 14.847 |
| asr_timings | 11.042 |
| ast_timings | 37.270 |
| describe_scenes | 20.821 |
| summarize_scenes | 24.192 |
| synthesize_synopsis | 18.326 |
| make_embedding | 4.790 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.430 |
| branch_yolo_total | 12.294 |
| branch_audio_total | 63.167 |
