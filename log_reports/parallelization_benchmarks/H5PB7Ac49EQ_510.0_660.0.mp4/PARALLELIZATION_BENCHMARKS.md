# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:29:40 UTC | H5PB7Ac49EQ_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 89.815 | 0.780 | 37.761 | 4.143 | 8.543 | 13.333 | 1.786 |

## 2026-06-25 02:29:40 UTC | H5PB7Ac49EQ_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/H5PB7Ac49EQ_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `89.815` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.298 |
| caption_frames | 13.687 |
| sample_fps | 1.842 |
| detect_object_yolo | 6.258 |
| audio_scan | 14.963 |
| asr_timings | 12.840 |
| ast_timings | 9.948 |
| describe_scenes | 4.143 |
| summarize_scenes | 8.543 |
| synthesize_synopsis | 13.333 |
| make_embedding | 1.786 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 13.990 |
| branch_yolo_total | 8.106 |
| branch_audio_total | 37.761 |
