# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:38:13 UTC | -cALRwbhf1E_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.977 | 0.801 | 45.690 | 13.976 | 26.551 | 19.595 | 3.151 |

## 2026-06-24 08:38:13 UTC | -cALRwbhf1E_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-cALRwbhf1E_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.977` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.801 |
| save_clips | - |
| sample_frames | 1.017 |
| caption_frames | 33.643 |
| sample_fps | 2.242 |
| detect_object_yolo | 7.962 |
| audio_scan | 12.809 |
| asr_timings | 8.474 |
| ast_timings | 24.398 |
| describe_scenes | 13.976 |
| summarize_scenes | 26.551 |
| synthesize_synopsis | 19.595 |
| make_embedding | 3.151 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.666 |
| branch_yolo_total | 10.210 |
| branch_audio_total | 45.690 |
