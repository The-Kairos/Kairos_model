# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:46:38 UTC | 00DH3yn5C30_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.575 | 1.583 | 67.804 | 4.511 | 4.903 | 5.543 | 0.944 |
| 2026-06-21 09:03:56 UTC | 00DH3yn5C30_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.058 | - | - | - | - | - | - |
| 2026-06-21 20:53:43 UTC | 00DH3yn5C30_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | - |
| 2026-06-22 12:15:27 UTC | 00DH3yn5C30_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.542 | 1.610 | 71.739 | 6.687 | 8.085 | 18.377 | 2.085 |

## 2026-06-19 22:46:38 UTC | 00DH3yn5C30_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.575` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.583 |
| save_clips | - |
| sample_frames | 0.944 |
| caption_frames | 16.463 |
| sample_fps | 5.166 |
| detect_object_yolo | 6.401 |
| audio_scan | 16.757 |
| asr_timings | 39.043 |
| ast_timings | 11.996 |
| describe_scenes | 4.511 |
| summarize_scenes | 4.903 |
| synthesize_synopsis | 5.543 |
| make_embedding | 0.944 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.413 |
| branch_yolo_total | 11.572 |
| branch_audio_total | 67.804 |

## 2026-06-21 09:03:56 UTC | 00DH3yn5C30_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `0.058` sec

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

## 2026-06-21 20:53:43 UTC | 00DH3yn5C30_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_660.0_810.0.mp4`
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

## 2026-06-22 12:15:27 UTC | 00DH3yn5C30_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.542` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.610 |
| save_clips | - |
| sample_frames | 0.902 |
| caption_frames | 17.044 |
| sample_fps | 5.191 |
| detect_object_yolo | 6.454 |
| audio_scan | 16.922 |
| asr_timings | 42.669 |
| ast_timings | 12.139 |
| describe_scenes | 6.687 |
| summarize_scenes | 8.085 |
| synthesize_synopsis | 18.377 |
| make_embedding | 2.085 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.952 |
| branch_yolo_total | 11.650 |
| branch_audio_total | 71.739 |
