# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:23:36 UTC | GxqR6p8r6z0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.851 | 0.673 | 68.282 | 13.434 | 15.035 | 12.794 | 5.300 |

## 2026-06-25 02:23:36 UTC | GxqR6p8r6z0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GxqR6p8r6z0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.851` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.673 |
| save_clips | - |
| sample_frames | 1.500 |
| caption_frames | 56.820 |
| sample_fps | 2.374 |
| detect_object_yolo | 11.215 |
| audio_scan | 11.748 |
| asr_timings | 11.930 |
| ast_timings | 44.597 |
| describe_scenes | 13.434 |
| summarize_scenes | 15.035 |
| synthesize_synopsis | 12.794 |
| make_embedding | 5.300 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 58.326 |
| branch_yolo_total | 13.596 |
| branch_audio_total | 68.282 |
