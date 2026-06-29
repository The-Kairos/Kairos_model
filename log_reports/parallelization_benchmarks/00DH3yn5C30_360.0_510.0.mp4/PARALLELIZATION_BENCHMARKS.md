# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:41:59 UTC | 00DH3yn5C30_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 77.323 | 1.611 | 33.975 | 3.795 | 4.820 | 5.308 | 0.842 |
| 2026-06-21 09:03:53 UTC | 00DH3yn5C30_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-21 20:53:40 UTC | 00DH3yn5C30_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:09:23 UTC | 00DH3yn5C30_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.757 | 1.591 | 35.118 | 7.736 | 6.237 | 34.820 | 1.839 |

## 2026-06-19 22:41:59 UTC | 00DH3yn5C30_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `77.323` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.611 |
| save_clips | - |
| sample_frames | 0.683 |
| caption_frames | 14.018 |
| sample_fps | 5.067 |
| detect_object_yolo | 5.884 |
| audio_scan | 14.581 |
| asr_timings | 9.563 |
| ast_timings | 9.822 |
| describe_scenes | 3.795 |
| summarize_scenes | 4.820 |
| synthesize_synopsis | 5.308 |
| make_embedding | 0.842 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.707 |
| branch_yolo_total | 10.957 |
| branch_audio_total | 33.975 |

## 2026-06-21 09:03:53 UTC | 00DH3yn5C30_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.059` sec

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

## 2026-06-21 20:53:40 UTC | 00DH3yn5C30_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_360.0_510.0.mp4`
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

## 2026-06-22 12:09:23 UTC | 00DH3yn5C30_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.757` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.591 |
| save_clips | - |
| sample_frames | 0.677 |
| caption_frames | 15.257 |
| sample_fps | 5.063 |
| detect_object_yolo | 6.040 |
| audio_scan | 14.816 |
| asr_timings | 10.215 |
| ast_timings | 10.078 |
| describe_scenes | 7.736 |
| summarize_scenes | 6.237 |
| synthesize_synopsis | 34.820 |
| make_embedding | 1.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.939 |
| branch_yolo_total | 11.109 |
| branch_audio_total | 35.118 |
