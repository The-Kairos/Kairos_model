# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:26:14 UTC | H5PB7Ac49EQ_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.646 | 0.787 | 58.420 | 12.007 | 18.435 | 8.090 | 3.839 |

## 2026-06-25 02:26:14 UTC | H5PB7Ac49EQ_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/H5PB7Ac49EQ_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.646` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.547 |
| caption_frames | 41.305 |
| sample_fps | 2.353 |
| detect_object_yolo | 9.445 |
| audio_scan | 15.909 |
| asr_timings | 10.852 |
| ast_timings | 31.650 |
| describe_scenes | 12.007 |
| summarize_scenes | 18.435 |
| synthesize_synopsis | 8.090 |
| make_embedding | 3.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.858 |
| branch_yolo_total | 11.804 |
| branch_audio_total | 58.420 |
