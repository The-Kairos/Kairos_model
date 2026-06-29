# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:59:04 UTC | fBFSes_K4u0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.234 | 0.782 | 68.447 | 5.896 | 17.565 | 9.458 | 2.560 |

## 2026-06-26 03:59:04 UTC | fBFSes_K4u0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fBFSes_K4u0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.234` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.647 |
| caption_frames | 24.482 |
| sample_fps | 2.036 |
| detect_object_yolo | 6.965 |
| audio_scan | 12.997 |
| asr_timings | 37.412 |
| ast_timings | 18.029 |
| describe_scenes | 5.896 |
| summarize_scenes | 17.565 |
| synthesize_synopsis | 9.458 |
| make_embedding | 2.560 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.135 |
| branch_yolo_total | 9.007 |
| branch_audio_total | 68.447 |
