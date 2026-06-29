# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:04:12 UTC | WlJGA2-wZQ4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 180.035 | 0.782 | 88.375 | 11.584 | 6.732 | 18.595 | 3.280 |

## 2026-06-25 21:04:12 UTC | WlJGA2-wZQ4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WlJGA2-wZQ4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.035` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 1.171 |
| caption_frames | 37.104 |
| sample_fps | 2.235 |
| detect_object_yolo | 8.760 |
| audio_scan | 14.959 |
| asr_timings | 45.903 |
| ast_timings | 27.505 |
| describe_scenes | 11.584 |
| summarize_scenes | 6.732 |
| synthesize_synopsis | 18.595 |
| make_embedding | 3.280 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.280 |
| branch_yolo_total | 11.001 |
| branch_audio_total | 88.375 |
