# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:17:01 UTC | GxqR6p8r6z0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 200.450 | 0.659 | 71.861 | 17.806 | 14.292 | 10.044 | 5.775 |

## 2026-06-25 02:17:01 UTC | GxqR6p8r6z0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GxqR6p8r6z0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `200.450` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.659 |
| save_clips | - |
| sample_frames | 1.576 |
| caption_frames | 62.121 |
| sample_fps | 2.495 |
| detect_object_yolo | 12.336 |
| audio_scan | 12.949 |
| asr_timings | 11.613 |
| ast_timings | 47.290 |
| describe_scenes | 17.806 |
| summarize_scenes | 14.292 |
| synthesize_synopsis | 10.044 |
| make_embedding | 5.775 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 63.703 |
| branch_yolo_total | 14.837 |
| branch_audio_total | 71.861 |
