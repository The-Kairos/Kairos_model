# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 18:01:16 UTC | m8802lFkHOs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.542 | 0.805 | 64.768 | 31.791 | 14.706 | 18.508 | 5.050 |

## 2026-06-26 18:01:16 UTC | m8802lFkHOs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/m8802lFkHOs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.542` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.341 |
| caption_frames | 56.643 |
| sample_fps | 2.445 |
| detect_object_yolo | 11.065 |
| audio_scan | 14.090 |
| asr_timings | 10.367 |
| ast_timings | 40.303 |
| describe_scenes | 31.791 |
| summarize_scenes | 14.706 |
| synthesize_synopsis | 18.508 |
| make_embedding | 5.050 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.990 |
| branch_yolo_total | 13.516 |
| branch_audio_total | 64.768 |
