# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-23 15:07:37 UTC | 2zUqHvqw0_8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 212.190 | 0.661 | 58.511 | 31.168 | 18.747 | 40.268 | 4.153 |
| 2026-06-24 09:07:30 UTC | 2zUqHvqw0_8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.641 | 0.672 | 58.771 | 20.853 | 32.037 | 19.356 | 4.154 |

## 2026-06-23 15:07:37 UTC | 2zUqHvqw0_8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `212.190` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 1.500 |
| caption_frames | 44.127 |
| sample_fps | 2.273 |
| detect_object_yolo | 9.389 |
| audio_scan | 14.799 |
| asr_timings | 10.023 |
| ast_timings | 33.680 |
| describe_scenes | 31.168 |
| summarize_scenes | 18.747 |
| synthesize_synopsis | 40.268 |
| make_embedding | 4.153 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.632 |
| branch_yolo_total | 11.668 |
| branch_audio_total | 58.511 |

## 2026-06-24 09:07:30 UTC | 2zUqHvqw0_8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/2zUqHvqw0_8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.641` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.672 |
| save_clips | - |
| sample_frames | 1.497 |
| caption_frames | 45.194 |
| sample_fps | 2.288 |
| detect_object_yolo | 9.431 |
| audio_scan | 14.942 |
| asr_timings | 9.975 |
| ast_timings | 33.846 |
| describe_scenes | 20.853 |
| summarize_scenes | 32.037 |
| synthesize_synopsis | 19.356 |
| make_embedding | 4.154 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.696 |
| branch_yolo_total | 11.724 |
| branch_audio_total | 58.771 |
