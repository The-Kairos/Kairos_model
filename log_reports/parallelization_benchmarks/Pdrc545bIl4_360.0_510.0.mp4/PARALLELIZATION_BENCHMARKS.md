# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:17:56 UTC | Pdrc545bIl4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.701 | 0.823 | 59.028 | 20.587 | 13.934 | 22.547 | 4.292 |

## 2026-06-25 14:17:56 UTC | Pdrc545bIl4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Pdrc545bIl4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.701` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.823 |
| save_clips | - |
| sample_frames | 1.329 |
| caption_frames | 48.005 |
| sample_fps | 2.460 |
| detect_object_yolo | 10.244 |
| audio_scan | 14.774 |
| asr_timings | 9.467 |
| ast_timings | 34.778 |
| describe_scenes | 20.587 |
| summarize_scenes | 13.934 |
| synthesize_synopsis | 22.547 |
| make_embedding | 4.292 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.340 |
| branch_yolo_total | 12.710 |
| branch_audio_total | 59.028 |
