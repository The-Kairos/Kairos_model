# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:57:48 UTC | S8ZUWshc1C4_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.266 | 0.638 | 57.101 | 12.270 | 28.221 | 20.091 | 2.787 |

## 2026-06-25 16:57:48 UTC | S8ZUWshc1C4_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/S8ZUWshc1C4_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.266` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.638 |
| save_clips | - |
| sample_frames | 0.933 |
| caption_frames | 30.231 |
| sample_fps | 1.984 |
| detect_object_yolo | 8.597 |
| audio_scan | 14.687 |
| asr_timings | 20.939 |
| ast_timings | 21.466 |
| describe_scenes | 12.270 |
| summarize_scenes | 28.221 |
| synthesize_synopsis | 20.091 |
| make_embedding | 2.787 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.170 |
| branch_yolo_total | 10.586 |
| branch_audio_total | 57.101 |
