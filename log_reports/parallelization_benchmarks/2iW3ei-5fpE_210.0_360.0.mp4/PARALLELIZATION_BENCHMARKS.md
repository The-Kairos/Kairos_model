# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:51:19 UTC | 2iW3ei-5fpE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 109.562 | 1.431 | 42.475 | 5.100 | 7.149 | 7.542 | 2.865 |
| 2026-06-21 21:30:31 UTC | 2iW3ei-5fpE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 106.371 | 1.403 | 42.866 | 5.396 | 5.175 | 5.979 | 2.825 |

## 2026-06-21 09:51:19 UTC | 2iW3ei-5fpE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2iW3ei-5fpE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `109.562` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.431 |
| save_clips | - |
| sample_frames | 1.384 |
| caption_frames | 27.980 |
| sample_fps | 4.862 |
| detect_object_yolo | 7.396 |
| audio_scan | 12.862 |
| asr_timings | 8.977 |
| ast_timings | 20.628 |
| describe_scenes | 5.100 |
| summarize_scenes | 7.149 |
| synthesize_synopsis | 7.542 |
| make_embedding | 2.865 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.370 |
| branch_yolo_total | 12.264 |
| branch_audio_total | 42.475 |

## 2026-06-21 21:30:31 UTC | 2iW3ei-5fpE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2iW3ei-5fpE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `106.371` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.403 |
| save_clips | - |
| sample_frames | 1.371 |
| caption_frames | 27.711 |
| sample_fps | 4.845 |
| detect_object_yolo | 7.417 |
| audio_scan | 12.896 |
| asr_timings | 9.005 |
| ast_timings | 20.957 |
| describe_scenes | 5.396 |
| summarize_scenes | 5.175 |
| synthesize_synopsis | 5.979 |
| make_embedding | 2.825 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.087 |
| branch_yolo_total | 12.268 |
| branch_audio_total | 42.866 |
