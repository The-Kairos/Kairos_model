# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:36:28 UTC | 4G9S-eoLgX4_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.868 | 0.704 | 46.017 | 6.201 | 6.132 | 6.426 | 2.795 |

## 2026-06-21 22:36:28 UTC | 4G9S-eoLgX4_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/4G9S-eoLgX4_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.868` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.704 |
| save_clips | - |
| sample_frames | 0.750 |
| caption_frames | 28.447 |
| sample_fps | 1.931 |
| detect_object_yolo | 7.066 |
| audio_scan | 14.943 |
| asr_timings | 9.867 |
| ast_timings | 21.199 |
| describe_scenes | 6.201 |
| summarize_scenes | 6.132 |
| synthesize_synopsis | 6.426 |
| make_embedding | 2.795 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.203 |
| branch_yolo_total | 9.003 |
| branch_audio_total | 46.017 |
