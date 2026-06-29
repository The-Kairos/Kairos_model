# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:26:40 UTC | iJJprA7tnLA_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.710 | 0.728 | 59.470 | 22.690 | 16.590 | 32.216 | 4.192 |

## 2026-06-26 08:26:40 UTC | iJJprA7tnLA_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iJJprA7tnLA_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.710` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.728 |
| save_clips | - |
| sample_frames | 1.144 |
| caption_frames | 44.743 |
| sample_fps | 2.112 |
| detect_object_yolo | 9.420 |
| audio_scan | 14.910 |
| asr_timings | 9.480 |
| ast_timings | 35.072 |
| describe_scenes | 22.690 |
| summarize_scenes | 16.590 |
| synthesize_synopsis | 32.216 |
| make_embedding | 4.192 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.893 |
| branch_yolo_total | 11.538 |
| branch_audio_total | 59.470 |
