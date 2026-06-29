# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 09:44:22 UTC | 2X46BBkcCeY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.843 | 1.951 | 58.864 | 12.158 | 10.347 | 6.339 | 5.058 |
| 2026-06-21 21:23:25 UTC | 2X46BBkcCeY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.648 | 1.974 | 59.537 | 12.074 | 8.675 | 10.958 | 5.055 |

## 2026-06-21 09:44:22 UTC | 2X46BBkcCeY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2X46BBkcCeY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.843` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.951 |
| save_clips | - |
| sample_frames | 4.263 |
| caption_frames | 51.470 |
| sample_fps | 6.803 |
| detect_object_yolo | 10.256 |
| audio_scan | 8.448 |
| asr_timings | 10.101 |
| ast_timings | 40.306 |
| describe_scenes | 12.158 |
| summarize_scenes | 10.347 |
| synthesize_synopsis | 6.339 |
| make_embedding | 5.058 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.740 |
| branch_yolo_total | 17.065 |
| branch_audio_total | 58.864 |

## 2026-06-21 21:23:25 UTC | 2X46BBkcCeY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2X46BBkcCeY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.648` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.974 |
| save_clips | - |
| sample_frames | 4.271 |
| caption_frames | 52.155 |
| sample_fps | 6.869 |
| detect_object_yolo | 10.686 |
| audio_scan | 8.575 |
| asr_timings | 10.316 |
| ast_timings | 40.638 |
| describe_scenes | 12.074 |
| summarize_scenes | 8.675 |
| synthesize_synopsis | 10.958 |
| make_embedding | 5.055 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.432 |
| branch_yolo_total | 17.560 |
| branch_audio_total | 59.537 |
