# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 09:06:07 UTC | i_2ND2tpKAc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.694 | 0.783 | 75.096 | 7.671 | 7.345 | 22.099 | 2.077 |

## 2026-06-26 09:06:07 UTC | i_2ND2tpKAc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i_2ND2tpKAc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.694` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 0.466 |
| caption_frames | 18.540 |
| sample_fps | 1.902 |
| detect_object_yolo | 6.322 |
| audio_scan | 16.159 |
| asr_timings | 45.766 |
| ast_timings | 13.162 |
| describe_scenes | 7.671 |
| summarize_scenes | 7.345 |
| synthesize_synopsis | 22.099 |
| make_embedding | 2.077 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.012 |
| branch_yolo_total | 8.230 |
| branch_audio_total | 75.096 |
