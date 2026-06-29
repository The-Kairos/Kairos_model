# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:01:42 UTC | 8fe7w1cAazA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.967 | 0.796 | 55.498 | 26.028 | 31.714 | 15.107 | 4.986 |

## 2026-06-24 17:01:42 UTC | 8fe7w1cAazA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8fe7w1cAazA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.967` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.464 |
| caption_frames | 54.028 |
| sample_fps | 2.462 |
| detect_object_yolo | 10.098 |
| audio_scan | 3.865 |
| asr_timings | 0.000 |
| ast_timings | 41.005 |
| describe_scenes | 26.028 |
| summarize_scenes | 31.714 |
| synthesize_synopsis | 15.107 |
| make_embedding | 4.986 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.498 |
| branch_yolo_total | 12.565 |
| branch_audio_total | 44.878 |
