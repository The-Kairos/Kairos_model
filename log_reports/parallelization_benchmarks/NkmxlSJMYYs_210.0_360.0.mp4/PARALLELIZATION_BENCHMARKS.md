# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:44:41 UTC | NkmxlSJMYYs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 97.130 | 0.792 | 30.217 | 8.731 | 20.086 | 18.365 | 1.383 |

## 2026-06-25 10:44:41 UTC | NkmxlSJMYYs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/NkmxlSJMYYs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `97.130` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 0.103 |
| caption_frames | 8.649 |
| sample_fps | 1.766 |
| detect_object_yolo | 5.620 |
| audio_scan | 15.385 |
| asr_timings | 10.179 |
| ast_timings | 4.644 |
| describe_scenes | 8.731 |
| summarize_scenes | 20.086 |
| synthesize_synopsis | 18.365 |
| make_embedding | 1.383 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.757 |
| branch_yolo_total | 7.392 |
| branch_audio_total | 30.217 |
