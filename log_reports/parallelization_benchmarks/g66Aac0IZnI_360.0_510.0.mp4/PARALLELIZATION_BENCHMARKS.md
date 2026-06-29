# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:51:39 UTC | g66Aac0IZnI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.256 | 0.784 | 57.006 | 10.551 | 12.552 | 14.951 | 3.937 |

## 2026-06-26 04:51:39 UTC | g66Aac0IZnI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/g66Aac0IZnI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.256` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.784 |
| save_clips | - |
| sample_frames | 1.200 |
| caption_frames | 41.146 |
| sample_fps | 2.317 |
| detect_object_yolo | 9.373 |
| audio_scan | 13.181 |
| asr_timings | 11.313 |
| ast_timings | 32.501 |
| describe_scenes | 10.551 |
| summarize_scenes | 12.552 |
| synthesize_synopsis | 14.951 |
| make_embedding | 3.937 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.352 |
| branch_yolo_total | 11.696 |
| branch_audio_total | 57.006 |
