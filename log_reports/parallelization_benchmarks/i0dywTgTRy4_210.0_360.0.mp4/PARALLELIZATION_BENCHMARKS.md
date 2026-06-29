# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:43:13 UTC | i0dywTgTRy4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.917 | 0.692 | 36.480 | 9.982 | 5.447 | 36.079 | 2.412 |

## 2026-06-26 07:43:13 UTC | i0dywTgTRy4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/i0dywTgTRy4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.917` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.692 |
| save_clips | - |
| sample_frames | 0.718 |
| caption_frames | 22.311 |
| sample_fps | 1.892 |
| detect_object_yolo | 7.449 |
| audio_scan | 11.944 |
| asr_timings | 9.362 |
| ast_timings | 15.165 |
| describe_scenes | 9.982 |
| summarize_scenes | 5.447 |
| synthesize_synopsis | 36.079 |
| make_embedding | 2.412 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.035 |
| branch_yolo_total | 9.347 |
| branch_audio_total | 36.480 |
