# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:01:50 UTC | _fLfMBVGhZY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.249 | 0.796 | 51.601 | 10.034 | 20.512 | 10.223 | 3.271 |

## 2026-06-26 00:01:50 UTC | _fLfMBVGhZY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_fLfMBVGhZY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.249` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.076 |
| caption_frames | 35.690 |
| sample_fps | 2.271 |
| detect_object_yolo | 8.350 |
| audio_scan | 12.867 |
| asr_timings | 11.057 |
| ast_timings | 27.668 |
| describe_scenes | 10.034 |
| summarize_scenes | 20.512 |
| synthesize_synopsis | 10.223 |
| make_embedding | 3.271 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.772 |
| branch_yolo_total | 10.627 |
| branch_audio_total | 51.601 |
