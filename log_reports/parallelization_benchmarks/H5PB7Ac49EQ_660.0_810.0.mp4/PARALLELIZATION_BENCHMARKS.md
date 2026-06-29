# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:33:34 UTC | H5PB7Ac49EQ_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 69.717 | 0.786 | 31.340 | 4.669 | 4.478 | 10.322 | 1.287 |

## 2026-06-25 02:33:34 UTC | H5PB7Ac49EQ_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/H5PB7Ac49EQ_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `69.717` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 0.107 |
| caption_frames | 7.790 |
| sample_fps | 1.771 |
| detect_object_yolo | 5.790 |
| audio_scan | 13.844 |
| asr_timings | 13.093 |
| ast_timings | 4.395 |
| describe_scenes | 4.669 |
| summarize_scenes | 4.478 |
| synthesize_synopsis | 10.322 |
| make_embedding | 1.287 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 7.903 |
| branch_yolo_total | 7.566 |
| branch_audio_total | 31.340 |
