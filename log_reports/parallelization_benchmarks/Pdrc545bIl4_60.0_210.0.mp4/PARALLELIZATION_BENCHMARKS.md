# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 14:21:09 UTC | Pdrc545bIl4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.957 | 0.795 | 62.947 | 23.168 | 12.986 | 21.822 | 4.723 |

## 2026-06-25 14:21:09 UTC | Pdrc545bIl4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Pdrc545bIl4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.957` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 1.419 |
| caption_frames | 49.817 |
| sample_fps | 2.460 |
| detect_object_yolo | 10.350 |
| audio_scan | 15.673 |
| asr_timings | 9.758 |
| ast_timings | 37.508 |
| describe_scenes | 23.168 |
| summarize_scenes | 12.986 |
| synthesize_synopsis | 21.822 |
| make_embedding | 4.723 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.242 |
| branch_yolo_total | 12.815 |
| branch_audio_total | 62.947 |
