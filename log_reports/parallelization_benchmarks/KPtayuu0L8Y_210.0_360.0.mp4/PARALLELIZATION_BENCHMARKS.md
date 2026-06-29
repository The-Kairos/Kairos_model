# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:30:35 UTC | KPtayuu0L8Y_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.883 | 0.748 | 38.946 | 10.236 | 10.664 | 16.779 | 2.319 |

## 2026-06-25 06:30:35 UTC | KPtayuu0L8Y_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KPtayuu0L8Y_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.883` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.748 |
| save_clips | - |
| sample_frames | 0.669 |
| caption_frames | 23.107 |
| sample_fps | 1.916 |
| detect_object_yolo | 7.112 |
| audio_scan | 14.868 |
| asr_timings | 8.150 |
| ast_timings | 15.920 |
| describe_scenes | 10.236 |
| summarize_scenes | 10.664 |
| synthesize_synopsis | 16.779 |
| make_embedding | 2.319 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.782 |
| branch_yolo_total | 9.033 |
| branch_audio_total | 38.946 |
