# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:52:13 UTC | G-4tJ63X5vo_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.382 | 0.645 | 69.971 | 15.844 | 11.880 | 17.085 | 5.064 |

## 2026-06-25 00:52:13 UTC | G-4tJ63X5vo_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G-4tJ63X5vo_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.382` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.645 |
| save_clips | - |
| sample_frames | 1.063 |
| caption_frames | 33.273 |
| sample_fps | 2.139 |
| detect_object_yolo | 10.084 |
| audio_scan | 17.449 |
| asr_timings | 11.338 |
| ast_timings | 41.175 |
| describe_scenes | 15.844 |
| summarize_scenes | 11.880 |
| synthesize_synopsis | 17.085 |
| make_embedding | 5.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.341 |
| branch_yolo_total | 12.229 |
| branch_audio_total | 69.971 |
