# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:13:21 UTC | 7YEQ0iDR4sw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.881 | 0.683 | 45.057 | 26.334 | 48.209 | 26.594 | 3.360 |

## 2026-06-24 14:13:21 UTC | 7YEQ0iDR4sw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7YEQ0iDR4sw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.881` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.683 |
| save_clips | - |
| sample_frames | 1.129 |
| caption_frames | 36.524 |
| sample_fps | 2.155 |
| detect_object_yolo | 8.372 |
| audio_scan | 8.616 |
| asr_timings | 9.415 |
| ast_timings | 27.017 |
| describe_scenes | 26.334 |
| summarize_scenes | 48.209 |
| synthesize_synopsis | 26.594 |
| make_embedding | 3.360 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.659 |
| branch_yolo_total | 10.532 |
| branch_audio_total | 45.057 |
