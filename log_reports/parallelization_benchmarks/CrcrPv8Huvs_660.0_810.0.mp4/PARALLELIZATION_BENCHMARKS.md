# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 21:04:56 UTC | CrcrPv8Huvs_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 68.958 | 0.662 | 28.259 | 4.710 | 3.380 | 9.561 | 1.563 |

## 2026-06-24 21:04:56 UTC | CrcrPv8Huvs_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CrcrPv8Huvs_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `68.958` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.662 |
| save_clips | - |
| sample_frames | 0.255 |
| caption_frames | 11.954 |
| sample_fps | 1.688 |
| detect_object_yolo | 5.544 |
| audio_scan | 13.814 |
| asr_timings | 7.580 |
| ast_timings | 6.857 |
| describe_scenes | 4.710 |
| summarize_scenes | 3.380 |
| synthesize_synopsis | 9.561 |
| make_embedding | 1.563 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 12.215 |
| branch_yolo_total | 7.238 |
| branch_audio_total | 28.259 |
