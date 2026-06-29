# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 10:10:16 UTC | 3Bk5MJEo2EA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 126.125 | 1.628 | 48.072 | 7.797 | 7.153 | 7.030 | 3.323 |
| 2026-06-21 21:49:39 UTC | 3Bk5MJEo2EA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 128.073 | 1.703 | 48.113 | 6.876 | 4.728 | 10.415 | 3.317 |

## 2026-06-21 10:10:16 UTC | 3Bk5MJEo2EA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `126.125` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.628 |
| save_clips | - |
| sample_frames | 2.615 |
| caption_frames | 33.696 |
| sample_fps | 5.375 |
| detect_object_yolo | 8.111 |
| audio_scan | 11.858 |
| asr_timings | 9.819 |
| ast_timings | 26.387 |
| describe_scenes | 7.797 |
| summarize_scenes | 7.153 |
| synthesize_synopsis | 7.030 |
| make_embedding | 3.323 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.317 |
| branch_yolo_total | 13.492 |
| branch_audio_total | 48.072 |

## 2026-06-21 21:49:39 UTC | 3Bk5MJEo2EA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3Bk5MJEo2EA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `128.073` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.703 |
| save_clips | - |
| sample_frames | 2.645 |
| caption_frames | 35.098 |
| sample_fps | 5.362 |
| detect_object_yolo | 8.419 |
| audio_scan | 11.840 |
| asr_timings | 9.368 |
| ast_timings | 26.897 |
| describe_scenes | 6.876 |
| summarize_scenes | 4.728 |
| synthesize_synopsis | 10.415 |
| make_embedding | 3.317 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.749 |
| branch_yolo_total | 13.787 |
| branch_audio_total | 48.113 |
