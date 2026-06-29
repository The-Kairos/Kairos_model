# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:56:38 UTC | l6aGqD9b53w_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.159 | 0.811 | 50.661 | 16.168 | 25.236 | 17.363 | 3.451 |

## 2026-06-26 14:56:38 UTC | l6aGqD9b53w_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l6aGqD9b53w_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.159` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 0.820 |
| caption_frames | 37.556 |
| sample_fps | 2.162 |
| detect_object_yolo | 8.533 |
| audio_scan | 12.929 |
| asr_timings | 10.124 |
| ast_timings | 27.600 |
| describe_scenes | 16.168 |
| summarize_scenes | 25.236 |
| synthesize_synopsis | 17.363 |
| make_embedding | 3.451 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.382 |
| branch_yolo_total | 10.700 |
| branch_audio_total | 50.661 |
