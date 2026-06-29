# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:05:15 UTC | iCbjk3uRtqA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.747 | 0.776 | 51.929 | 20.981 | 36.691 | 33.608 | 3.460 |

## 2026-06-26 08:05:15 UTC | iCbjk3uRtqA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iCbjk3uRtqA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.747` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.776 |
| save_clips | - |
| sample_frames | 0.987 |
| caption_frames | 38.229 |
| sample_fps | 2.187 |
| detect_object_yolo | 8.478 |
| audio_scan | 15.164 |
| asr_timings | 9.267 |
| ast_timings | 27.489 |
| describe_scenes | 20.981 |
| summarize_scenes | 36.691 |
| synthesize_synopsis | 33.608 |
| make_embedding | 3.460 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.221 |
| branch_yolo_total | 10.671 |
| branch_audio_total | 51.929 |
