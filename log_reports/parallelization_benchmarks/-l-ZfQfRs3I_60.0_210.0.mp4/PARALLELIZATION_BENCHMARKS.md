# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-19 22:40:41 UTC | -l-ZfQfRs3I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.484 | 1.908 | 67.618 | 13.836 | 15.936 | 6.807 | 2.106 |
| 2026-06-21 09:03:52 UTC | -l-ZfQfRs3I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-21 20:53:37 UTC | -l-ZfQfRs3I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 12:02:14 UTC | -l-ZfQfRs3I_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 227.400 | 1.929 | 68.717 | 27.182 | 23.830 | 24.078 | 5.008 |

## 2026-06-19 22:40:41 UTC | -l-ZfQfRs3I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-l-ZfQfRs3I_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.484` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.908 |
| save_clips | - |
| sample_frames | 4.245 |
| caption_frames | 51.453 |
| sample_fps | 6.001 |
| detect_object_yolo | 10.234 |
| audio_scan | 16.821 |
| asr_timings | 10.904 |
| ast_timings | 39.885 |
| describe_scenes | 13.836 |
| summarize_scenes | 15.936 |
| synthesize_synopsis | 6.807 |
| make_embedding | 2.106 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.704 |
| branch_yolo_total | 16.241 |
| branch_audio_total | 67.618 |

## 2026-06-21 09:03:52 UTC | -l-ZfQfRs3I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-l-ZfQfRs3I_60.0_210.0.mp4`
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

## 2026-06-21 20:53:37 UTC | -l-ZfQfRs3I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-l-ZfQfRs3I_60.0_210.0.mp4`
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

## 2026-06-22 12:02:14 UTC | -l-ZfQfRs3I_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-l-ZfQfRs3I_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `227.400` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.929 |
| save_clips | - |
| sample_frames | 4.291 |
| caption_frames | 54.505 |
| sample_fps | 5.981 |
| detect_object_yolo | 10.496 |
| audio_scan | 16.927 |
| asr_timings | 10.662 |
| ast_timings | 41.120 |
| describe_scenes | 27.182 |
| summarize_scenes | 23.830 |
| synthesize_synopsis | 24.078 |
| make_embedding | 5.008 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.802 |
| branch_yolo_total | 16.484 |
| branch_audio_total | 68.717 |
