# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:44:41 UTC | 00DH3yn5C30_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 76.987 | 1.584 | 34.866 | 7.520 | 4.290 | 5.965 | 0.771 |
| 2026-06-21 09:03:55 UTC | 00DH3yn5C30_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-21 20:53:42 UTC | 00DH3yn5C30_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | - |
| 2026-06-22 12:13:07 UTC | 00DH3yn5C30_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.992 | 1.560 | 34.429 | 5.491 | 24.253 | 27.140 | 1.544 |

## 2026-06-19 22:44:41 UTC | 00DH3yn5C30_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `76.987` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.584 |
| save_clips | - |
| sample_frames | 0.374 |
| caption_frames | 9.796 |
| sample_fps | 4.932 |
| detect_object_yolo | 5.578 |
| audio_scan | 15.757 |
| asr_timings | 12.097 |
| ast_timings | 7.004 |
| describe_scenes | 7.520 |
| summarize_scenes | 4.290 |
| synthesize_synopsis | 5.965 |
| make_embedding | 0.771 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.175 |
| branch_yolo_total | 10.516 |
| branch_audio_total | 34.866 |

## 2026-06-21 09:03:55 UTC | 00DH3yn5C30_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_60.0_210.0.mp4`
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

## 2026-06-21 20:53:42 UTC | 00DH3yn5C30_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.061` sec

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

## 2026-06-22 12:13:07 UTC | 00DH3yn5C30_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.992` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.560 |
| save_clips | - |
| sample_frames | 0.389 |
| caption_frames | 10.119 |
| sample_fps | 4.967 |
| detect_object_yolo | 5.713 |
| audio_scan | 15.941 |
| asr_timings | 11.372 |
| ast_timings | 7.108 |
| describe_scenes | 5.491 |
| summarize_scenes | 24.253 |
| synthesize_synopsis | 27.140 |
| make_embedding | 1.544 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.513 |
| branch_yolo_total | 10.686 |
| branch_audio_total | 34.429 |
