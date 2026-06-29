# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:18:01 UTC | ja7n-uhv9bg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.328 | 0.788 | 57.442 | 21.030 | 16.743 | 17.086 | 3.920 |

## 2026-06-26 11:18:01 UTC | ja7n-uhv9bg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ja7n-uhv9bg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.328` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 1.247 |
| caption_frames | 44.018 |
| sample_fps | 2.357 |
| detect_object_yolo | 9.294 |
| audio_scan | 14.058 |
| asr_timings | 10.194 |
| ast_timings | 33.181 |
| describe_scenes | 21.030 |
| summarize_scenes | 16.743 |
| synthesize_synopsis | 17.086 |
| make_embedding | 3.920 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.271 |
| branch_yolo_total | 11.656 |
| branch_audio_total | 57.442 |
