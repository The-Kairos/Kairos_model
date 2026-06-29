# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 08:00:46 UTC | M-zRrwcpfMg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 220.228 | 0.775 | 75.457 | 22.550 | 30.006 | 17.374 | 4.573 |

## 2026-06-25 08:00:46 UTC | M-zRrwcpfMg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/M-zRrwcpfMg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `220.228` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.775 |
| save_clips | - |
| sample_frames | 1.640 |
| caption_frames | 53.472 |
| sample_fps | 2.570 |
| detect_object_yolo | 10.342 |
| audio_scan | 16.054 |
| asr_timings | 22.052 |
| ast_timings | 37.342 |
| describe_scenes | 22.550 |
| summarize_scenes | 30.006 |
| synthesize_synopsis | 17.374 |
| make_embedding | 4.573 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.118 |
| branch_yolo_total | 12.918 |
| branch_audio_total | 75.457 |
