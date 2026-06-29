# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 16:50:43 UTC | nwgLQy1b_Ro_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.085 | 0.824 | 50.007 | 12.296 | 11.796 | 7.815 | 3.280 |

## 2026-06-27 16:50:43 UTC | nwgLQy1b_Ro_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/nwgLQy1b_Ro_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.085` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.824 |
| save_clips | - |
| sample_frames | 1.286 |
| caption_frames | 37.903 |
| sample_fps | 2.290 |
| detect_object_yolo | 8.199 |
| audio_scan | 14.842 |
| asr_timings | 7.862 |
| ast_timings | 27.294 |
| describe_scenes | 12.296 |
| summarize_scenes | 11.796 |
| synthesize_synopsis | 7.815 |
| make_embedding | 3.280 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.195 |
| branch_yolo_total | 10.495 |
| branch_audio_total | 50.007 |
