# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:38 UTC | -oPEG4FKHy8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.059 | - | - | - | - | - | - |
| 2026-06-22 12:05:13 UTC | -oPEG4FKHy8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.314 | 1.265 | 96.317 | 5.818 | 23.385 | 19.750 | 2.064 |

## 2026-06-21 20:53:38 UTC | -oPEG4FKHy8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-oPEG4FKHy8_60.0_210.0.mp4`
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

## 2026-06-22 12:05:13 UTC | -oPEG4FKHy8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-oPEG4FKHy8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.314` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.265 |
| save_clips | - |
| sample_frames | 0.666 |
| caption_frames | 18.495 |
| sample_fps | 1.864 |
| detect_object_yolo | 6.285 |
| audio_scan | 15.910 |
| asr_timings | 67.816 |
| ast_timings | 12.581 |
| describe_scenes | 5.818 |
| summarize_scenes | 23.385 |
| synthesize_synopsis | 19.750 |
| make_embedding | 2.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.167 |
| branch_yolo_total | 8.155 |
| branch_audio_total | 96.317 |
