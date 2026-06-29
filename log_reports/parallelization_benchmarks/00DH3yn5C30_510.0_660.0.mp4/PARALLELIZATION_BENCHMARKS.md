# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:43:23 UTC | 00DH3yn5C30_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 82.260 | 1.567 | 37.536 | 4.170 | 4.345 | 6.357 | 0.873 |
| 2026-06-21 09:03:54 UTC | 00DH3yn5C30_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-21 20:53:41 UTC | 00DH3yn5C30_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:11:09 UTC | 00DH3yn5C30_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 104.202 | 1.570 | 37.145 | 7.596 | 9.848 | 17.581 | 1.795 |

## 2026-06-19 22:43:23 UTC | 00DH3yn5C30_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `82.260` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.567 |
| save_clips | - |
| sample_frames | 0.675 |
| caption_frames | 14.259 |
| sample_fps | 5.088 |
| detect_object_yolo | 6.070 |
| audio_scan | 15.669 |
| asr_timings | 11.711 |
| ast_timings | 10.147 |
| describe_scenes | 4.170 |
| summarize_scenes | 4.345 |
| synthesize_synopsis | 6.357 |
| make_embedding | 0.873 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.940 |
| branch_yolo_total | 11.164 |
| branch_audio_total | 37.536 |

## 2026-06-21 09:03:54 UTC | 00DH3yn5C30_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_510.0_660.0.mp4`
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

## 2026-06-21 20:53:41 UTC | 00DH3yn5C30_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_510.0_660.0.mp4`
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

## 2026-06-22 12:11:09 UTC | 00DH3yn5C30_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/00DH3yn5C30_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `104.202` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.570 |
| save_clips | - |
| sample_frames | 0.639 |
| caption_frames | 15.239 |
| sample_fps | 5.100 |
| detect_object_yolo | 6.287 |
| audio_scan | 15.998 |
| asr_timings | 11.035 |
| ast_timings | 10.103 |
| describe_scenes | 7.596 |
| summarize_scenes | 9.848 |
| synthesize_synopsis | 17.581 |
| make_embedding | 1.795 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.884 |
| branch_yolo_total | 11.393 |
| branch_audio_total | 37.145 |
