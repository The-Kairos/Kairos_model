# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:30:51 UTC | MQ1bAV_vIRI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.495 | 0.783 | 52.896 | 12.222 | 30.020 | 20.215 | 3.036 |

## 2026-06-25 09:30:51 UTC | MQ1bAV_vIRI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MQ1bAV_vIRI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.495` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.783 |
| save_clips | - |
| sample_frames | 0.683 |
| caption_frames | 29.811 |
| sample_fps | 2.069 |
| detect_object_yolo | 7.337 |
| audio_scan | 5.332 |
| asr_timings | 26.335 |
| ast_timings | 21.220 |
| describe_scenes | 12.222 |
| summarize_scenes | 30.020 |
| synthesize_synopsis | 20.215 |
| make_embedding | 3.036 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.499 |
| branch_yolo_total | 9.412 |
| branch_audio_total | 52.896 |
