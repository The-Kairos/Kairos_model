# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:11:53 UTC | 13U4xVzZFQ8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.996 | 1.824 | 62.383 | 13.123 | 8.929 | 11.067 | 4.723 |
| 2026-06-21 20:54:09 UTC | 13U4xVzZFQ8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 13:21:35 UTC | 13U4xVzZFQ8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 234.075 | 1.795 | 63.618 | 26.768 | 41.226 | 22.575 | 4.768 |

## 2026-06-21 09:11:53 UTC | 13U4xVzZFQ8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.996` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.824 |
| save_clips | - |
| sample_frames | 3.543 |
| caption_frames | 48.343 |
| sample_fps | 6.587 |
| detect_object_yolo | 10.155 |
| audio_scan | 15.748 |
| asr_timings | 8.989 |
| ast_timings | 37.634 |
| describe_scenes | 13.123 |
| summarize_scenes | 8.929 |
| synthesize_synopsis | 11.067 |
| make_embedding | 4.723 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.892 |
| branch_yolo_total | 16.748 |
| branch_audio_total | 62.383 |

## 2026-06-21 20:54:09 UTC | 13U4xVzZFQ8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.060` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | - |
| save_clips | - |
| sample_frames | - |
| caption_frames | - |
| sample_fps | - |
| detect_object_yolo | - |
| audio_scan | - |
| asr_timings | - |
| ast_timings | - |
| describe_scenes | - |
| summarize_scenes | - |
| synthesize_synopsis | - |
| make_embedding | - |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | - |
| branch_yolo_total | - |
| branch_audio_total | - |

## 2026-06-22 13:21:35 UTC | 13U4xVzZFQ8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/13U4xVzZFQ8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `234.075` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.795 |
| save_clips | - |
| sample_frames | 3.588 |
| caption_frames | 51.142 |
| sample_fps | 6.669 |
| detect_object_yolo | 10.526 |
| audio_scan | 16.137 |
| asr_timings | 9.172 |
| ast_timings | 38.301 |
| describe_scenes | 26.768 |
| summarize_scenes | 41.226 |
| synthesize_synopsis | 22.575 |
| make_embedding | 4.768 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.736 |
| branch_yolo_total | 17.201 |
| branch_audio_total | 63.618 |
