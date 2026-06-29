# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:42:02 UTC | 4G9S-eoLgX4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.416 | 0.686 | 66.433 | 13.082 | 11.957 | 7.293 | 5.008 |

## 2026-06-21 22:42:02 UTC | 4G9S-eoLgX4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4G9S-eoLgX4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.416` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.686 |
| save_clips | - |
| sample_frames | 1.802 |
| caption_frames | 58.924 |
| sample_fps | 2.443 |
| detect_object_yolo | 10.378 |
| audio_scan | 16.024 |
| asr_timings | 8.966 |
| ast_timings | 41.434 |
| describe_scenes | 13.082 |
| summarize_scenes | 11.957 |
| synthesize_synopsis | 7.293 |
| make_embedding | 5.008 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.732 |
| branch_yolo_total | 12.827 |
| branch_audio_total | 66.433 |
