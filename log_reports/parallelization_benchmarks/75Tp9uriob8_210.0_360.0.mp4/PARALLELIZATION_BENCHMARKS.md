# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 13:25:31 UTC | 75Tp9uriob8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.417 | 0.676 | 50.980 | 19.075 | 17.358 | 25.775 | 3.811 |

## 2026-06-24 13:25:31 UTC | 75Tp9uriob8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/75Tp9uriob8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.417` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.676 |
| save_clips | - |
| sample_frames | 1.074 |
| caption_frames | 26.142 |
| sample_fps | 2.097 |
| detect_object_yolo | 8.409 |
| audio_scan | 12.987 |
| asr_timings | 7.918 |
| ast_timings | 30.067 |
| describe_scenes | 19.075 |
| summarize_scenes | 17.358 |
| synthesize_synopsis | 25.775 |
| make_embedding | 3.811 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.221 |
| branch_yolo_total | 10.511 |
| branch_audio_total | 50.980 |
