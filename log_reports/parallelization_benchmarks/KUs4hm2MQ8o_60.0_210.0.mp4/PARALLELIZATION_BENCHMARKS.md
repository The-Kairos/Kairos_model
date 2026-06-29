# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:42:31 UTC | KUs4hm2MQ8o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 137.976 | 0.679 | 42.083 | 12.834 | 9.424 | 33.887 | 2.544 |

## 2026-06-25 06:42:31 UTC | KUs4hm2MQ8o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KUs4hm2MQ8o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `137.976` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.679 |
| save_clips | - |
| sample_frames | 0.569 |
| caption_frames | 25.368 |
| sample_fps | 1.866 |
| detect_object_yolo | 7.334 |
| audio_scan | 10.547 |
| asr_timings | 12.947 |
| ast_timings | 18.580 |
| describe_scenes | 12.834 |
| summarize_scenes | 9.424 |
| synthesize_synopsis | 33.887 |
| make_embedding | 2.544 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.943 |
| branch_yolo_total | 9.207 |
| branch_audio_total | 42.083 |
