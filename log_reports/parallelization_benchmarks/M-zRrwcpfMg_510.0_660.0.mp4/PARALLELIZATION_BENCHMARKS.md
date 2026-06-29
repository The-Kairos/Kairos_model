# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 08:07:37 UTC | M-zRrwcpfMg_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 209.573 | 0.814 | 86.073 | 22.249 | 16.751 | 14.732 | 4.175 |

## 2026-06-25 08:07:37 UTC | M-zRrwcpfMg_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/M-zRrwcpfMg_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `209.573` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 1.662 |
| caption_frames | 49.927 |
| sample_fps | 2.558 |
| detect_object_yolo | 9.166 |
| audio_scan | 13.831 |
| asr_timings | 38.835 |
| ast_timings | 33.398 |
| describe_scenes | 22.249 |
| summarize_scenes | 16.751 |
| synthesize_synopsis | 14.732 |
| make_embedding | 4.175 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.594 |
| branch_yolo_total | 11.730 |
| branch_audio_total | 86.073 |
