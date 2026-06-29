# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:17:31 UTC | 2zUqHvqw0_8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 211.110 | 0.641 | 63.692 | 28.965 | 20.632 | 27.672 | 4.724 |
| 2026-06-24 09:16:51 UTC | 2zUqHvqw0_8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.394 | 0.636 | 64.046 | 23.703 | 13.786 | 20.926 | 4.723 |

## 2026-06-23 15:17:31 UTC | 2zUqHvqw0_8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `211.110` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.641 |
| save_clips | - |
| sample_frames | 1.496 |
| caption_frames | 49.370 |
| sample_fps | 2.255 |
| detect_object_yolo | 10.302 |
| audio_scan | 15.825 |
| asr_timings | 10.497 |
| ast_timings | 37.362 |
| describe_scenes | 28.965 |
| summarize_scenes | 20.632 |
| synthesize_synopsis | 27.672 |
| make_embedding | 4.724 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.872 |
| branch_yolo_total | 12.562 |
| branch_audio_total | 63.692 |

## 2026-06-24 09:16:51 UTC | 2zUqHvqw0_8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.394` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.636 |
| save_clips | - |
| sample_frames | 1.492 |
| caption_frames | 50.101 |
| sample_fps | 2.274 |
| detect_object_yolo | 10.315 |
| audio_scan | 15.883 |
| asr_timings | 10.547 |
| ast_timings | 37.608 |
| describe_scenes | 23.703 |
| summarize_scenes | 13.786 |
| synthesize_synopsis | 20.926 |
| make_embedding | 4.723 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.598 |
| branch_yolo_total | 12.595 |
| branch_audio_total | 64.046 |
