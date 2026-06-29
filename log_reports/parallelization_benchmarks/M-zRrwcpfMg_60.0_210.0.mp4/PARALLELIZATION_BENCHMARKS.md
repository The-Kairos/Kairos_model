# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 08:10:55 UTC | M-zRrwcpfMg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 196.983 | 0.781 | 68.083 | 22.581 | 24.247 | 23.946 | 3.639 |

## 2026-06-25 08:10:55 UTC | M-zRrwcpfMg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/M-zRrwcpfMg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `196.983` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.130 |
| caption_frames | 40.338 |
| sample_fps | 2.303 |
| detect_object_yolo | 8.521 |
| audio_scan | 14.730 |
| asr_timings | 24.322 |
| ast_timings | 29.021 |
| describe_scenes | 22.581 |
| summarize_scenes | 24.247 |
| synthesize_synopsis | 23.946 |
| make_embedding | 3.639 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.474 |
| branch_yolo_total | 10.830 |
| branch_audio_total | 68.083 |
