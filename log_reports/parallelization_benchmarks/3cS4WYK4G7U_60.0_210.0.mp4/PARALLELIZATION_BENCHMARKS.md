# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 16:29:45 UTC | 3cS4WYK4G7U_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.634 | 0.766 | 58.625 | 26.831 | 17.712 | 16.272 | 4.260 |
| 2026-06-24 10:25:08 UTC | 3cS4WYK4G7U_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.175 | 0.771 | 58.540 | 20.055 | 11.720 | 17.410 | 4.219 |

## 2026-06-23 16:29:45 UTC | 3cS4WYK4G7U_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.634` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 1.368 |
| caption_frames | 46.430 |
| sample_fps | 2.340 |
| detect_object_yolo | 9.663 |
| audio_scan | 13.833 |
| asr_timings | 10.618 |
| ast_timings | 34.165 |
| describe_scenes | 26.831 |
| summarize_scenes | 17.712 |
| synthesize_synopsis | 16.272 |
| make_embedding | 4.260 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.805 |
| branch_yolo_total | 12.009 |
| branch_audio_total | 58.625 |

## 2026-06-24 10:25:08 UTC | 3cS4WYK4G7U_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3cS4WYK4G7U_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.175` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.771 |
| save_clips | - |
| sample_frames | 1.386 |
| caption_frames | 47.491 |
| sample_fps | 2.409 |
| detect_object_yolo | 9.767 |
| audio_scan | 13.870 |
| asr_timings | 10.526 |
| ast_timings | 34.135 |
| describe_scenes | 20.055 |
| summarize_scenes | 11.720 |
| synthesize_synopsis | 17.410 |
| make_embedding | 4.219 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.883 |
| branch_yolo_total | 12.182 |
| branch_audio_total | 58.540 |
