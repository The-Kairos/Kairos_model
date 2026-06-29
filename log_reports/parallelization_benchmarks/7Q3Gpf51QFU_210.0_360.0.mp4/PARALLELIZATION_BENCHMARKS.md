# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:58:36 UTC | 7Q3Gpf51QFU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 221.648 | 0.814 | 69.849 | 29.733 | 20.512 | 23.603 | 5.354 |

## 2026-06-24 13:58:36 UTC | 7Q3Gpf51QFU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7Q3Gpf51QFU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `221.648` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 1.578 |
| caption_frames | 55.123 |
| sample_fps | 2.548 |
| detect_object_yolo | 11.127 |
| audio_scan | 14.981 |
| asr_timings | 11.919 |
| ast_timings | 42.940 |
| describe_scenes | 29.733 |
| summarize_scenes | 20.512 |
| synthesize_synopsis | 23.603 |
| make_embedding | 5.354 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.707 |
| branch_yolo_total | 13.681 |
| branch_audio_total | 69.849 |
