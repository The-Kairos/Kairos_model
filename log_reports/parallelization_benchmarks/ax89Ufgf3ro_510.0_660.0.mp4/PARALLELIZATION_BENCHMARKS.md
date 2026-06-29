# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:32:20 UTC | ax89Ufgf3ro_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.563 | 0.663 | 57.830 | 11.158 | 7.587 | 10.039 | 3.247 |

## 2026-06-26 00:32:20 UTC | ax89Ufgf3ro_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ax89Ufgf3ro_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.563` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.663 |
| save_clips | - |
| sample_frames | 1.161 |
| caption_frames | 34.830 |
| sample_fps | 2.095 |
| detect_object_yolo | 8.468 |
| audio_scan | 15.081 |
| asr_timings | 15.966 |
| ast_timings | 26.774 |
| describe_scenes | 11.158 |
| summarize_scenes | 7.587 |
| synthesize_synopsis | 10.039 |
| make_embedding | 3.247 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.997 |
| branch_yolo_total | 10.569 |
| branch_audio_total | 57.830 |
