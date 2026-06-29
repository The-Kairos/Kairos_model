# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:31:34 UTC | POU_6XcdD1s_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 190.151 | 0.630 | 78.025 | 19.451 | 8.382 | 40.205 | 2.820 |

## 2026-06-25 13:31:34 UTC | POU_6XcdD1s_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/POU_6XcdD1s_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `190.151` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.630 |
| save_clips | - |
| sample_frames | 0.720 |
| caption_frames | 28.845 |
| sample_fps | 1.916 |
| detect_object_yolo | 7.766 |
| audio_scan | 15.488 |
| asr_timings | 41.018 |
| ast_timings | 21.510 |
| describe_scenes | 19.451 |
| summarize_scenes | 8.382 |
| synthesize_synopsis | 40.205 |
| make_embedding | 2.820 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.571 |
| branch_yolo_total | 9.687 |
| branch_audio_total | 78.025 |
