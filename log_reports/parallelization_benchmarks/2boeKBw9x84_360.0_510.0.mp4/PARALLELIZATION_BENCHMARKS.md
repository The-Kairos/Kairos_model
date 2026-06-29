# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 14:53:24 UTC | 2boeKBw9x84_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.930 | 0.730 | 40.773 | 28.642 | 16.913 | 24.101 | 3.265 |
| 2026-06-24 08:54:20 UTC | 2boeKBw9x84_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 152.772 | 0.732 | 41.584 | 22.887 | 14.880 | 18.891 | 3.396 |

## 2026-06-23 14:53:24 UTC | 2boeKBw9x84_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boeKBw9x84_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.930` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.730 |
| save_clips | - |
| sample_frames | 1.418 |
| caption_frames | 37.266 |
| sample_fps | 2.241 |
| detect_object_yolo | 8.211 |
| audio_scan | 5.333 |
| asr_timings | 8.247 |
| ast_timings | 27.185 |
| describe_scenes | 28.642 |
| summarize_scenes | 16.913 |
| synthesize_synopsis | 24.101 |
| make_embedding | 3.265 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.689 |
| branch_yolo_total | 10.457 |
| branch_audio_total | 40.773 |

## 2026-06-24 08:54:20 UTC | 2boeKBw9x84_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2boeKBw9x84_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `152.772` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.732 |
| save_clips | - |
| sample_frames | 1.436 |
| caption_frames | 36.918 |
| sample_fps | 2.283 |
| detect_object_yolo | 8.355 |
| audio_scan | 5.311 |
| asr_timings | 9.022 |
| ast_timings | 27.242 |
| describe_scenes | 22.887 |
| summarize_scenes | 14.880 |
| synthesize_synopsis | 18.891 |
| make_embedding | 3.396 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.361 |
| branch_yolo_total | 10.643 |
| branch_audio_total | 41.584 |
