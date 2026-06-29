# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:12:10 UTC | fMActfTJboI_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.609 | 0.830 | 67.084 | 14.310 | 12.490 | 7.798 | 4.433 |

## 2026-06-26 04:12:10 UTC | fMActfTJboI_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fMActfTJboI_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.609` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.830 |
| save_clips | - |
| sample_frames | 1.415 |
| caption_frames | 47.552 |
| sample_fps | 2.443 |
| detect_object_yolo | 9.810 |
| audio_scan | 16.362 |
| asr_timings | 13.874 |
| ast_timings | 36.840 |
| describe_scenes | 14.310 |
| summarize_scenes | 12.490 |
| synthesize_synopsis | 7.798 |
| make_embedding | 4.433 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.973 |
| branch_yolo_total | 12.258 |
| branch_audio_total | 67.084 |
