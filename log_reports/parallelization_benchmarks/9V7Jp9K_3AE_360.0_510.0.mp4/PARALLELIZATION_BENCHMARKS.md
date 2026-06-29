# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:01:04 UTC | 9V7Jp9K_3AE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 242.829 | 0.689 | 87.528 | 25.302 | 16.250 | 20.345 | 6.837 |

## 2026-06-24 18:01:04 UTC | 9V7Jp9K_3AE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9V7Jp9K_3AE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `242.829` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.689 |
| save_clips | - |
| sample_frames | 1.817 |
| caption_frames | 67.838 |
| sample_fps | 2.541 |
| detect_object_yolo | 12.256 |
| audio_scan | 8.620 |
| asr_timings | 25.866 |
| ast_timings | 53.033 |
| describe_scenes | 25.302 |
| summarize_scenes | 16.250 |
| synthesize_synopsis | 20.345 |
| make_embedding | 6.837 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 69.660 |
| branch_yolo_total | 14.802 |
| branch_audio_total | 87.528 |
