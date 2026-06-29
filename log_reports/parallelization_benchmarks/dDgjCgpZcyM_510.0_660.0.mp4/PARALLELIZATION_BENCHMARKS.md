# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:50:18 UTC | dDgjCgpZcyM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 84.684 | 0.651 | 35.241 | 6.082 | 5.570 | 12.167 | 1.781 |

## 2026-06-26 02:50:18 UTC | dDgjCgpZcyM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dDgjCgpZcyM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `84.684` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.651 |
| save_clips | - |
| sample_frames | 0.277 |
| caption_frames | 13.532 |
| sample_fps | 1.701 |
| detect_object_yolo | 6.293 |
| audio_scan | 15.120 |
| asr_timings | 9.973 |
| ast_timings | 10.140 |
| describe_scenes | 6.082 |
| summarize_scenes | 5.570 |
| synthesize_synopsis | 12.167 |
| make_embedding | 1.781 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.815 |
| branch_yolo_total | 8.000 |
| branch_audio_total | 35.241 |
