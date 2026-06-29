# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:32:24 UTC | x6QkZM27EVw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 146.273 | 0.656 | 81.143 | 6.280 | 8.875 | 9.372 | 2.518 |

## 2026-06-27 03:32:24 UTC | x6QkZM27EVw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/x6QkZM27EVw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `146.273` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 0.770 |
| caption_frames | 26.057 |
| sample_fps | 1.971 |
| detect_object_yolo | 7.213 |
| audio_scan | 13.028 |
| asr_timings | 49.564 |
| ast_timings | 18.542 |
| describe_scenes | 6.280 |
| summarize_scenes | 8.875 |
| synthesize_synopsis | 9.372 |
| make_embedding | 2.518 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.832 |
| branch_yolo_total | 9.189 |
| branch_audio_total | 81.143 |
