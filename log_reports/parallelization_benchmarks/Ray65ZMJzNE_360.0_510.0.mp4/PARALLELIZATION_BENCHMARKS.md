# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:38:46 UTC | Ray65ZMJzNE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.690 | 0.803 | 59.591 | 7.878 | 10.147 | 22.743 | 2.108 |

## 2026-06-25 16:38:46 UTC | Ray65ZMJzNE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ray65ZMJzNE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.690` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 0.537 |
| caption_frames | 18.628 |
| sample_fps | 2.006 |
| detect_object_yolo | 6.822 |
| audio_scan | 11.183 |
| asr_timings | 35.343 |
| ast_timings | 13.055 |
| describe_scenes | 7.878 |
| summarize_scenes | 10.147 |
| synthesize_synopsis | 22.743 |
| make_embedding | 2.108 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.171 |
| branch_yolo_total | 8.834 |
| branch_audio_total | 59.591 |
