# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:58:28 UTC | 8fe7w1cAazA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 108.384 | 0.951 | 29.673 | 17.296 | 7.723 | 13.648 | 3.100 |

## 2026-06-24 16:58:28 UTC | 8fe7w1cAazA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8fe7w1cAazA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `108.384` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.951 |
| save_clips | - |
| sample_frames | 1.288 |
| caption_frames | 28.380 |
| sample_fps | 2.300 |
| detect_object_yolo | 6.854 |
| audio_scan | 3.850 |
| asr_timings | 0.000 |
| ast_timings | 21.586 |
| describe_scenes | 17.296 |
| summarize_scenes | 7.723 |
| synthesize_synopsis | 13.648 |
| make_embedding | 3.100 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.673 |
| branch_yolo_total | 9.160 |
| branch_audio_total | 25.444 |
