# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:17:54 UTC | 1rguOCdMIY8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.536 | 1.363 | 69.169 | 13.586 | 9.998 | 9.258 | 5.394 |
| 2026-06-21 20:54:11 UTC | 1rguOCdMIY8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.061 | - | - | - | - | - | - |
| 2026-06-22 13:29:55 UTC | 1rguOCdMIY8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 241.017 | 1.364 | 70.137 | 31.235 | 20.345 | 30.405 | 5.440 |

## 2026-06-21 09:17:54 UTC | 1rguOCdMIY8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1rguOCdMIY8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.536` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.363 |
| save_clips | - |
| sample_frames | 3.655 |
| caption_frames | 56.880 |
| sample_fps | 6.269 |
| detect_object_yolo | 10.651 |
| audio_scan | 15.871 |
| asr_timings | 10.356 |
| ast_timings | 42.933 |
| describe_scenes | 13.586 |
| summarize_scenes | 9.998 |
| synthesize_synopsis | 9.258 |
| make_embedding | 5.394 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.541 |
| branch_yolo_total | 16.925 |
| branch_audio_total | 69.169 |

## 2026-06-21 20:54:11 UTC | 1rguOCdMIY8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1rguOCdMIY8_210.0_360.0.mp4`
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

## 2026-06-22 13:29:55 UTC | 1rguOCdMIY8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/1rguOCdMIY8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `241.017` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.364 |
| save_clips | - |
| sample_frames | 3.720 |
| caption_frames | 59.513 |
| sample_fps | 6.345 |
| detect_object_yolo | 11.124 |
| audio_scan | 15.962 |
| asr_timings | 10.617 |
| ast_timings | 43.549 |
| describe_scenes | 31.235 |
| summarize_scenes | 20.345 |
| synthesize_synopsis | 30.405 |
| make_embedding | 5.440 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.239 |
| branch_yolo_total | 17.475 |
| branch_audio_total | 70.137 |
