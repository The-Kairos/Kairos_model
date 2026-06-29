# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:07:14 UTC | jZnB2yjitMs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 195.859 | 0.802 | 64.592 | 20.032 | 15.382 | 23.939 | 4.826 |

## 2026-06-26 11:07:14 UTC | jZnB2yjitMs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jZnB2yjitMs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `195.859` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.174 |
| caption_frames | 50.638 |
| sample_fps | 2.379 |
| detect_object_yolo | 10.681 |
| audio_scan | 14.040 |
| asr_timings | 12.288 |
| ast_timings | 38.255 |
| describe_scenes | 20.032 |
| summarize_scenes | 15.382 |
| synthesize_synopsis | 23.939 |
| make_embedding | 4.826 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.818 |
| branch_yolo_total | 13.066 |
| branch_audio_total | 64.592 |
