# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:45:59 UTC | Pt5YUleXhTg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 278.771 | 0.804 | 65.605 | 23.261 | 74.190 | 41.681 | 5.159 |

## 2026-06-25 14:45:59 UTC | Pt5YUleXhTg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Pt5YUleXhTg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `278.771` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 1.588 |
| caption_frames | 51.780 |
| sample_fps | 2.532 |
| detect_object_yolo | 10.626 |
| audio_scan | 12.263 |
| asr_timings | 13.280 |
| ast_timings | 40.053 |
| describe_scenes | 23.261 |
| summarize_scenes | 74.190 |
| synthesize_synopsis | 41.681 |
| make_embedding | 5.159 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.375 |
| branch_yolo_total | 13.164 |
| branch_audio_total | 65.605 |
