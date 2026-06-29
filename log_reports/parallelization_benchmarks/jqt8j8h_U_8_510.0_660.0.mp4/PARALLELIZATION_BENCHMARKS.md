# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:16:58 UTC | jqt8j8h_U_8_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 339.201 | 0.685 | 264.073 | 8.939 | 20.023 | 14.071 | 2.086 |

## 2026-06-26 12:16:58 UTC | jqt8j8h_U_8_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jqt8j8h_U_8_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `339.201` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.685 |
| save_clips | - |
| sample_frames | 0.588 |
| caption_frames | 18.531 |
| sample_fps | 1.867 |
| detect_object_yolo | 6.930 |
| audio_scan | 13.988 |
| asr_timings | 236.780 |
| ast_timings | 13.296 |
| describe_scenes | 8.939 |
| summarize_scenes | 20.023 |
| synthesize_synopsis | 14.071 |
| make_embedding | 2.086 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.124 |
| branch_yolo_total | 8.803 |
| branch_audio_total | 264.073 |
