# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:16:23 UTC | FQEW3xLOa9M_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 97.914 | 0.779 | 29.622 | 6.437 | 8.820 | 14.211 | 2.748 |

## 2026-06-25 00:16:23 UTC | FQEW3xLOa9M_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/FQEW3xLOa9M_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `97.914` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 0.657 |
| caption_frames | 28.959 |
| sample_fps | 2.026 |
| detect_object_yolo | 6.851 |
| audio_scan | 3.879 |
| asr_timings | 0.000 |
| ast_timings | 21.132 |
| describe_scenes | 6.437 |
| summarize_scenes | 8.820 |
| synthesize_synopsis | 14.211 |
| make_embedding | 2.748 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.622 |
| branch_yolo_total | 8.883 |
| branch_audio_total | 25.020 |
