# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 04:54:57 UTC | yPyNZtMGZTk_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.687 | 0.659 | 46.893 | 7.430 | 7.399 | 7.007 | 2.765 |

## 2026-06-27 04:54:57 UTC | yPyNZtMGZTk_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yPyNZtMGZTk_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.687` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.659 |
| save_clips | - |
| sample_frames | 0.664 |
| caption_frames | 30.828 |
| sample_fps | 1.951 |
| detect_object_yolo | 7.680 |
| audio_scan | 15.201 |
| asr_timings | 10.050 |
| ast_timings | 21.634 |
| describe_scenes | 7.430 |
| summarize_scenes | 7.399 |
| synthesize_synopsis | 7.007 |
| make_embedding | 2.765 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.498 |
| branch_yolo_total | 9.637 |
| branch_audio_total | 46.893 |
