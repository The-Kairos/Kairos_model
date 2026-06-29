# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:35:06 UTC | 2YGUY_4d8D0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 259.526 | 0.790 | 70.344 | 32.665 | 32.645 | 31.824 | 6.451 |
| 2026-06-27 15:50:08 UTC | 2YGUY_4d8D0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.361 | 0.803 | 71.396 | 14.001 | 14.605 | 6.022 | 6.474 |

## 2026-06-23 14:35:06 UTC | 2YGUY_4d8D0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2YGUY_4d8D0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `259.526` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 1.948 |
| caption_frames | 66.277 |
| sample_fps | 2.888 |
| detect_object_yolo | 12.302 |
| audio_scan | 7.404 |
| asr_timings | 11.877 |
| ast_timings | 51.055 |
| describe_scenes | 32.665 |
| summarize_scenes | 32.645 |
| synthesize_synopsis | 31.824 |
| make_embedding | 6.451 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 68.231 |
| branch_yolo_total | 15.196 |
| branch_audio_total | 70.344 |

## 2026-06-27 15:50:08 UTC | 2YGUY_4d8D0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2YGUY_4d8D0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.361` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.972 |
| caption_frames | 67.211 |
| sample_fps | 2.894 |
| detect_object_yolo | 12.578 |
| audio_scan | 7.575 |
| asr_timings | 12.309 |
| ast_timings | 51.504 |
| describe_scenes | 14.001 |
| summarize_scenes | 14.605 |
| synthesize_synopsis | 6.022 |
| make_embedding | 6.474 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 69.188 |
| branch_yolo_total | 15.478 |
| branch_audio_total | 71.396 |
