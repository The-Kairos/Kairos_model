# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:37:58 UTC | KUs4hm2MQ8o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.986 | 0.643 | 44.030 | 12.058 | 43.944 | 20.963 | 3.353 |

## 2026-06-25 06:37:58 UTC | KUs4hm2MQ8o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KUs4hm2MQ8o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.986` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 0.750 |
| caption_frames | 35.213 |
| sample_fps | 2.011 |
| detect_object_yolo | 8.635 |
| audio_scan | 9.640 |
| asr_timings | 9.915 |
| ast_timings | 24.466 |
| describe_scenes | 12.058 |
| summarize_scenes | 43.944 |
| synthesize_synopsis | 20.963 |
| make_embedding | 3.353 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.969 |
| branch_yolo_total | 10.651 |
| branch_audio_total | 44.030 |
