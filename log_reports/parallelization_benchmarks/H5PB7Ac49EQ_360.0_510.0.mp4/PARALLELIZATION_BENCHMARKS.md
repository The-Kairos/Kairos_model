# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 02:28:10 UTC | H5PB7Ac49EQ_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.025 | 0.782 | 45.927 | 9.786 | 6.781 | 11.238 | 2.529 |

## 2026-06-25 02:28:10 UTC | H5PB7Ac49EQ_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/H5PB7Ac49EQ_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.025` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.782 |
| save_clips | - |
| sample_frames | 0.880 |
| caption_frames | 25.305 |
| sample_fps | 2.104 |
| detect_object_yolo | 7.284 |
| audio_scan | 16.002 |
| asr_timings | 11.558 |
| ast_timings | 18.358 |
| describe_scenes | 9.786 |
| summarize_scenes | 6.781 |
| synthesize_synopsis | 11.238 |
| make_embedding | 2.529 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.192 |
| branch_yolo_total | 9.394 |
| branch_audio_total | 45.927 |
