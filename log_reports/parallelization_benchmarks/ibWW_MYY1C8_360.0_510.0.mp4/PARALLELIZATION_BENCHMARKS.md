# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:16:57 UTC | ibWW_MYY1C8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 203.173 | 0.694 | 83.442 | 19.628 | 13.249 | 22.725 | 3.912 |

## 2026-06-26 09:16:57 UTC | ibWW_MYY1C8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ibWW_MYY1C8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `203.173` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.694 |
| save_clips | - |
| sample_frames | 1.556 |
| caption_frames | 44.548 |
| sample_fps | 2.311 |
| detect_object_yolo | 9.669 |
| audio_scan | 12.943 |
| asr_timings | 37.412 |
| ast_timings | 33.078 |
| describe_scenes | 19.628 |
| summarize_scenes | 13.249 |
| synthesize_synopsis | 22.725 |
| make_embedding | 3.912 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.110 |
| branch_yolo_total | 11.986 |
| branch_audio_total | 83.442 |
