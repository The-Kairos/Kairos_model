# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:30:49 UTC | _QWKqFFxaw8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.680 | 0.655 | 72.481 | 11.639 | 6.755 | 9.395 | 3.549 |

## 2026-06-25 23:30:49 UTC | _QWKqFFxaw8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_QWKqFFxaw8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.680` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.655 |
| save_clips | - |
| sample_frames | 0.866 |
| caption_frames | 38.052 |
| sample_fps | 2.044 |
| detect_object_yolo | 8.810 |
| audio_scan | 15.229 |
| asr_timings | 26.951 |
| ast_timings | 30.293 |
| describe_scenes | 11.639 |
| summarize_scenes | 6.755 |
| synthesize_synopsis | 9.395 |
| make_embedding | 3.549 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.924 |
| branch_yolo_total | 10.859 |
| branch_audio_total | 72.481 |
