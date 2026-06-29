# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:28:04 UTC | xv_YH57C1MU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.459 | 0.684 | 61.083 | 14.333 | 10.452 | 8.586 | 4.448 |

## 2026-06-27 04:28:04 UTC | xv_YH57C1MU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xv_YH57C1MU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.459` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 1.671 |
| caption_frames | 49.196 |
| sample_fps | 2.348 |
| detect_object_yolo | 10.260 |
| audio_scan | 15.145 |
| asr_timings | 8.913 |
| ast_timings | 37.016 |
| describe_scenes | 14.333 |
| summarize_scenes | 10.452 |
| synthesize_synopsis | 8.586 |
| make_embedding | 4.448 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.873 |
| branch_yolo_total | 12.614 |
| branch_audio_total | 61.083 |
