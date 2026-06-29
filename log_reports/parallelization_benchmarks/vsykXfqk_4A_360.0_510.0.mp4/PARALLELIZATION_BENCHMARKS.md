# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:38:02 UTC | vsykXfqk_4A_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.262 | 0.789 | 61.422 | 13.869 | 11.963 | 7.834 | 5.088 |

## 2026-06-27 02:38:02 UTC | vsykXfqk_4A_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vsykXfqk_4A_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.262` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.414 |
| caption_frames | 54.403 |
| sample_fps | 2.469 |
| detect_object_yolo | 10.600 |
| audio_scan | 10.748 |
| asr_timings | 8.688 |
| ast_timings | 41.978 |
| describe_scenes | 13.869 |
| summarize_scenes | 11.963 |
| synthesize_synopsis | 7.834 |
| make_embedding | 5.088 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.823 |
| branch_yolo_total | 13.075 |
| branch_audio_total | 61.422 |
