# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:57:07 UTC | 2vFzBa_WKNg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 111.890 | 3.398 | 41.041 | 5.549 | 5.929 | 7.921 | 2.536 |
| 2026-06-21 21:36:33 UTC | 2vFzBa_WKNg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 112.168 | 3.375 | 40.900 | 4.746 | 6.351 | 7.695 | 2.537 |

## 2026-06-21 09:57:07 UTC | 2vFzBa_WKNg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `111.890` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.398 |
| save_clips | - |
| sample_frames | 2.754 |
| caption_frames | 24.494 |
| sample_fps | 9.835 |
| detect_object_yolo | 7.096 |
| audio_scan | 7.610 |
| asr_timings | 15.057 |
| ast_timings | 18.366 |
| describe_scenes | 5.549 |
| summarize_scenes | 5.929 |
| synthesize_synopsis | 7.921 |
| make_embedding | 2.536 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.253 |
| branch_yolo_total | 16.937 |
| branch_audio_total | 41.041 |

## 2026-06-21 21:36:33 UTC | 2vFzBa_WKNg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `112.168` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.375 |
| save_clips | - |
| sample_frames | 2.733 |
| caption_frames | 25.221 |
| sample_fps | 9.983 |
| detect_object_yolo | 7.250 |
| audio_scan | 7.579 |
| asr_timings | 14.508 |
| ast_timings | 18.804 |
| describe_scenes | 4.746 |
| summarize_scenes | 6.351 |
| synthesize_synopsis | 7.695 |
| make_embedding | 2.537 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.960 |
| branch_yolo_total | 17.238 |
| branch_audio_total | 40.900 |
