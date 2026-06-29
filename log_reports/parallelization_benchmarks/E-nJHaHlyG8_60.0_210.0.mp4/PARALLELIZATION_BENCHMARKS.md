# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:33:34 UTC | E-nJHaHlyG8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.888 | 0.887 | 63.356 | 11.369 | 9.186 | 10.063 | 5.131 |

## 2026-06-24 23:33:34 UTC | E-nJHaHlyG8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/E-nJHaHlyG8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.888` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.887 |
| save_clips | - |
| sample_frames | 1.581 |
| caption_frames | 54.462 |
| sample_fps | 2.685 |
| detect_object_yolo | 10.735 |
| audio_scan | 12.868 |
| asr_timings | 9.311 |
| ast_timings | 41.168 |
| describe_scenes | 11.369 |
| summarize_scenes | 9.186 |
| synthesize_synopsis | 10.063 |
| make_embedding | 5.131 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.049 |
| branch_yolo_total | 13.426 |
| branch_audio_total | 63.356 |
