# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:46:40 UTC | GPZ4So5mepU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.995 | 0.665 | 53.931 | 12.073 | 13.506 | 14.734 | 3.531 |

## 2026-06-25 01:46:40 UTC | GPZ4So5mepU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GPZ4So5mepU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.995` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.665 |
| save_clips | - |
| sample_frames | 1.034 |
| caption_frames | 38.691 |
| sample_fps | 2.113 |
| detect_object_yolo | 9.315 |
| audio_scan | 14.890 |
| asr_timings | 9.428 |
| ast_timings | 29.604 |
| describe_scenes | 12.073 |
| summarize_scenes | 13.506 |
| synthesize_synopsis | 14.734 |
| make_embedding | 3.531 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.731 |
| branch_yolo_total | 11.434 |
| branch_audio_total | 53.931 |
