# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 20:53:54 UTC | 0lbehz52PFU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 0.060 | - | - | - | - | - | - |
| 2026-06-22 12:40:46 UTC | 0lbehz52PFU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 217.115 | 0.766 | 68.978 | 24.175 | 25.284 | 17.593 | 5.454 |

## 2026-06-21 20:53:54 UTC | 0lbehz52PFU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0lbehz52PFU_210.0_360.0.mp4`
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

## 2026-06-22 12:40:46 UTC | 0lbehz52PFU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/0lbehz52PFU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `217.115` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.766 |
| save_clips | - |
| sample_frames | 1.402 |
| caption_frames | 58.794 |
| sample_fps | 2.446 |
| detect_object_yolo | 10.821 |
| audio_scan | 12.734 |
| asr_timings | 12.958 |
| ast_timings | 43.278 |
| describe_scenes | 24.175 |
| summarize_scenes | 25.284 |
| synthesize_synopsis | 17.593 |
| make_embedding | 5.454 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.202 |
| branch_yolo_total | 13.273 |
| branch_audio_total | 68.978 |
