# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:30:01 UTC | U2Svw1Y9X3E_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 193.537 | 0.805 | 67.251 | 13.914 | 23.423 | 14.399 | 4.983 |

## 2026-06-25 18:30:01 UTC | U2Svw1Y9X3E_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/U2Svw1Y9X3E_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `193.537` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.457 |
| caption_frames | 52.900 |
| sample_fps | 2.459 |
| detect_object_yolo | 10.541 |
| audio_scan | 11.717 |
| asr_timings | 14.220 |
| ast_timings | 41.305 |
| describe_scenes | 13.914 |
| summarize_scenes | 23.423 |
| synthesize_synopsis | 14.399 |
| make_embedding | 4.983 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.363 |
| branch_yolo_total | 13.005 |
| branch_audio_total | 67.251 |
