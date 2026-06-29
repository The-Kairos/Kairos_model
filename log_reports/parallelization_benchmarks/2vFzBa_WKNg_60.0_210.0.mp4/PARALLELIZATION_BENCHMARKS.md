# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:04:32 UTC | 2vFzBa_WKNg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 95.806 | 3.405 | 33.095 | 3.608 | 6.101 | 9.673 | 2.140 |
| 2026-06-21 21:43:58 UTC | 2vFzBa_WKNg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 96.695 | 3.543 | 33.587 | 3.670 | 5.957 | 7.941 | 2.083 |

## 2026-06-21 10:04:32 UTC | 2vFzBa_WKNg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `95.806` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.405 |
| save_clips | - |
| sample_frames | 2.280 |
| caption_frames | 18.038 |
| sample_fps | 9.582 |
| detect_object_yolo | 6.581 |
| audio_scan | 8.684 |
| asr_timings | 11.646 |
| ast_timings | 12.756 |
| describe_scenes | 3.608 |
| summarize_scenes | 6.101 |
| synthesize_synopsis | 9.673 |
| make_embedding | 2.140 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 20.324 |
| branch_yolo_total | 16.168 |
| branch_audio_total | 33.095 |

## 2026-06-21 21:43:58 UTC | 2vFzBa_WKNg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2vFzBa_WKNg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `96.695` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 3.543 |
| save_clips | - |
| sample_frames | 2.202 |
| caption_frames | 19.251 |
| sample_fps | 10.010 |
| detect_object_yolo | 6.992 |
| audio_scan | 8.818 |
| asr_timings | 11.762 |
| ast_timings | 12.998 |
| describe_scenes | 3.670 |
| summarize_scenes | 5.957 |
| synthesize_synopsis | 7.941 |
| make_embedding | 2.083 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.459 |
| branch_yolo_total | 17.007 |
| branch_audio_total | 33.587 |
