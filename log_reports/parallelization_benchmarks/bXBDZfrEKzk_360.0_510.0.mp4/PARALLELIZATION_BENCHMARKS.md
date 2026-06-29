# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:24:57 UTC | bXBDZfrEKzk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.216 | 0.811 | 47.587 | 8.878 | 5.128 | 12.330 | 3.138 |

## 2026-06-26 01:24:57 UTC | bXBDZfrEKzk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bXBDZfrEKzk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.216` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 0.782 |
| caption_frames | 20.265 |
| sample_fps | 2.070 |
| detect_object_yolo | 7.088 |
| audio_scan | 16.729 |
| asr_timings | 9.243 |
| ast_timings | 21.607 |
| describe_scenes | 8.878 |
| summarize_scenes | 5.128 |
| synthesize_synopsis | 12.330 |
| make_embedding | 3.138 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.053 |
| branch_yolo_total | 9.164 |
| branch_audio_total | 47.587 |
