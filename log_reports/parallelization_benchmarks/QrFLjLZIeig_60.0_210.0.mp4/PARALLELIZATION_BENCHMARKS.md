# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 15:39:14 UTC | QrFLjLZIeig_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.856 | 0.786 | 46.810 | 12.341 | 22.549 | 16.188 | 2.794 |

## 2026-06-25 15:39:14 UTC | QrFLjLZIeig_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QrFLjLZIeig_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.856` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.797 |
| caption_frames | 30.510 |
| sample_fps | 2.111 |
| detect_object_yolo | 7.560 |
| audio_scan | 15.536 |
| asr_timings | 10.187 |
| ast_timings | 21.079 |
| describe_scenes | 12.341 |
| summarize_scenes | 22.549 |
| synthesize_synopsis | 16.188 |
| make_embedding | 2.794 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.313 |
| branch_yolo_total | 9.677 |
| branch_audio_total | 46.810 |
