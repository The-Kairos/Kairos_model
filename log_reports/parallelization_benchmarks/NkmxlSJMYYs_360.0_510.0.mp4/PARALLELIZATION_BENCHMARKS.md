# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:46:24 UTC | NkmxlSJMYYs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 101.844 | 0.786 | 33.487 | 4.851 | 20.420 | 16.271 | 1.837 |

## 2026-06-25 10:46:24 UTC | NkmxlSJMYYs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/NkmxlSJMYYs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `101.844` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.308 |
| caption_frames | 14.982 |
| sample_fps | 1.826 |
| detect_object_yolo | 5.667 |
| audio_scan | 14.003 |
| asr_timings | 9.717 |
| ast_timings | 9.758 |
| describe_scenes | 4.851 |
| summarize_scenes | 20.420 |
| synthesize_synopsis | 16.271 |
| make_embedding | 1.837 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 15.296 |
| branch_yolo_total | 7.498 |
| branch_audio_total | 33.487 |
