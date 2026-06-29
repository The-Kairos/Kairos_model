# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:25:54 UTC | GDi-Pbip33M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.988 | 0.807 | 63.143 | 23.829 | 18.117 | 18.731 | 4.447 |

## 2026-06-25 01:25:54 UTC | GDi-Pbip33M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GDi-Pbip33M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.988` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.807 |
| save_clips | - |
| sample_frames | 1.676 |
| caption_frames | 48.988 |
| sample_fps | 2.600 |
| detect_object_yolo | 10.240 |
| audio_scan | 13.842 |
| asr_timings | 11.008 |
| ast_timings | 38.284 |
| describe_scenes | 23.829 |
| summarize_scenes | 18.117 |
| synthesize_synopsis | 18.731 |
| make_embedding | 4.447 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.670 |
| branch_yolo_total | 12.846 |
| branch_audio_total | 63.143 |
