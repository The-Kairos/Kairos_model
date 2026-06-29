# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 13:42:48 UTC | 1WSstSrReZQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 206.116 | 0.623 | 54.973 | 31.681 | 31.922 | 26.668 | 3.901 |
| 2026-06-27 15:13:20 UTC | 1WSstSrReZQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 147.332 | 0.652 | 55.342 | 12.570 | 7.828 | 9.591 | 3.887 |

## 2026-06-23 13:42:48 UTC | 1WSstSrReZQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1WSstSrReZQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `206.116` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.623 |
| save_clips | - |
| sample_frames | 1.210 |
| caption_frames | 43.455 |
| sample_fps | 2.034 |
| detect_object_yolo | 8.291 |
| audio_scan | 15.078 |
| asr_timings | 7.975 |
| ast_timings | 31.912 |
| describe_scenes | 31.681 |
| summarize_scenes | 31.922 |
| synthesize_synopsis | 26.668 |
| make_embedding | 3.901 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.670 |
| branch_yolo_total | 10.331 |
| branch_audio_total | 54.973 |

## 2026-06-27 15:13:20 UTC | 1WSstSrReZQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1WSstSrReZQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `147.332` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.652 |
| save_clips | - |
| sample_frames | 1.237 |
| caption_frames | 44.047 |
| sample_fps | 2.066 |
| detect_object_yolo | 8.672 |
| audio_scan | 15.405 |
| asr_timings | 7.679 |
| ast_timings | 32.249 |
| describe_scenes | 12.570 |
| summarize_scenes | 7.828 |
| synthesize_synopsis | 9.591 |
| make_embedding | 3.887 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.291 |
| branch_yolo_total | 10.744 |
| branch_audio_total | 55.342 |
