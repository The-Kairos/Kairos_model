# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:08:34 UTC | wR-XEd-TBJk_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.530 | 0.903 | 45.023 | 6.535 | 8.606 | 8.530 | 2.997 |

## 2026-06-27 03:08:34 UTC | wR-XEd-TBJk_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wR-XEd-TBJk_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.530` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.903 |
| save_clips | - |
| sample_frames | 0.823 |
| caption_frames | 29.823 |
| sample_fps | 2.195 |
| detect_object_yolo | 7.698 |
| audio_scan | 10.835 |
| asr_timings | 10.239 |
| ast_timings | 23.942 |
| describe_scenes | 6.535 |
| summarize_scenes | 8.606 |
| synthesize_synopsis | 8.530 |
| make_embedding | 2.997 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.652 |
| branch_yolo_total | 9.898 |
| branch_audio_total | 45.023 |
