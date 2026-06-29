# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:56:39 UTC | 8fe7w1cAazA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 102.489 | 0.808 | 21.129 | 8.393 | 18.180 | 22.206 | 2.294 |

## 2026-06-24 16:56:39 UTC | 8fe7w1cAazA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8fe7w1cAazA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `102.489` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.808 |
| save_clips | - |
| sample_frames | 0.550 |
| caption_frames | 20.573 |
| sample_fps | 1.975 |
| detect_object_yolo | 6.292 |
| audio_scan | 3.865 |
| asr_timings | 0.000 |
| ast_timings | 15.924 |
| describe_scenes | 8.393 |
| summarize_scenes | 18.180 |
| synthesize_synopsis | 22.206 |
| make_embedding | 2.294 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.129 |
| branch_yolo_total | 8.272 |
| branch_audio_total | 19.797 |
