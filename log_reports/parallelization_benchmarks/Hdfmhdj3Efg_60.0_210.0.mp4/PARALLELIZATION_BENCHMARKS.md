# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:52:35 UTC | Hdfmhdj3Efg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.117 | 0.901 | 47.527 | 13.755 | 6.825 | 7.605 | 3.940 |

## 2026-06-25 03:52:35 UTC | Hdfmhdj3Efg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Hdfmhdj3Efg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.117` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.901 |
| save_clips | - |
| sample_frames | 1.567 |
| caption_frames | 36.889 |
| sample_fps | 2.529 |
| detect_object_yolo | 8.168 |
| audio_scan | 12.756 |
| asr_timings | 7.331 |
| ast_timings | 27.432 |
| describe_scenes | 13.755 |
| summarize_scenes | 6.825 |
| synthesize_synopsis | 7.605 |
| make_embedding | 3.940 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.462 |
| branch_yolo_total | 10.702 |
| branch_audio_total | 47.527 |
