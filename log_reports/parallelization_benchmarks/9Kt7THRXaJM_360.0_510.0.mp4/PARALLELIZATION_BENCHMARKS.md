# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:45:31 UTC | 9Kt7THRXaJM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.841 | 0.794 | 63.185 | 18.803 | 14.632 | 16.225 | 4.960 |

## 2026-06-24 17:45:31 UTC | 9Kt7THRXaJM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9Kt7THRXaJM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.841` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.794 |
| save_clips | - |
| sample_frames | 1.322 |
| caption_frames | 51.493 |
| sample_fps | 2.426 |
| detect_object_yolo | 10.617 |
| audio_scan | 13.925 |
| asr_timings | 8.414 |
| ast_timings | 40.837 |
| describe_scenes | 18.803 |
| summarize_scenes | 14.632 |
| synthesize_synopsis | 16.225 |
| make_embedding | 4.960 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.821 |
| branch_yolo_total | 13.049 |
| branch_audio_total | 63.185 |
