# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:54:40 UTC | ChCi1CGFt50_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.470 | 0.723 | 63.588 | 14.822 | 9.918 | 10.347 | 4.962 |

## 2026-06-24 20:54:40 UTC | ChCi1CGFt50_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ChCi1CGFt50_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.470` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.723 |
| save_clips | - |
| sample_frames | 1.478 |
| caption_frames | 52.611 |
| sample_fps | 2.350 |
| detect_object_yolo | 10.247 |
| audio_scan | 15.029 |
| asr_timings | 8.242 |
| ast_timings | 40.309 |
| describe_scenes | 14.822 |
| summarize_scenes | 9.918 |
| synthesize_synopsis | 10.347 |
| make_embedding | 4.962 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.095 |
| branch_yolo_total | 12.604 |
| branch_audio_total | 63.588 |
