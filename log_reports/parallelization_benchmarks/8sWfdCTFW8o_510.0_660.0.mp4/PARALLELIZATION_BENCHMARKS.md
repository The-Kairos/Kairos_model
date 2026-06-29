# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:18:43 UTC | 8sWfdCTFW8o_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.750 | 0.693 | 51.731 | 22.146 | 19.252 | 26.834 | 3.710 |

## 2026-06-24 17:18:43 UTC | 8sWfdCTFW8o_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8sWfdCTFW8o_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.750` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.693 |
| save_clips | - |
| sample_frames | 1.033 |
| caption_frames | 43.495 |
| sample_fps | 2.119 |
| detect_object_yolo | 9.266 |
| audio_scan | 11.812 |
| asr_timings | 10.038 |
| ast_timings | 29.872 |
| describe_scenes | 22.146 |
| summarize_scenes | 19.252 |
| synthesize_synopsis | 26.834 |
| make_embedding | 3.710 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.533 |
| branch_yolo_total | 11.391 |
| branch_audio_total | 51.731 |
