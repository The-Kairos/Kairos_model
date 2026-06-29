# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 20:53:57 UTC | sk3p9-ynrNE_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.660 | 0.791 | 39.041 | 7.431 | 11.245 | 13.257 | 2.262 |

## 2026-06-26 20:53:57 UTC | sk3p9-ynrNE_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sk3p9-ynrNE_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.660` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 0.694 |
| caption_frames | 23.049 |
| sample_fps | 2.093 |
| detect_object_yolo | 7.304 |
| audio_scan | 14.119 |
| asr_timings | 8.963 |
| ast_timings | 15.950 |
| describe_scenes | 7.431 |
| summarize_scenes | 11.245 |
| synthesize_synopsis | 13.257 |
| make_embedding | 2.262 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.749 |
| branch_yolo_total | 9.403 |
| branch_audio_total | 39.041 |
