# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:10:24 UTC | 9_UC5c7LFN4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 104.871 | 0.780 | 28.962 | 9.434 | 23.578 | 14.576 | 1.823 |

## 2026-06-24 18:10:24 UTC | 9_UC5c7LFN4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9_UC5c7LFN4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `104.871` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 0.376 |
| caption_frames | 16.060 |
| sample_fps | 1.929 |
| detect_object_yolo | 5.973 |
| audio_scan | 6.511 |
| asr_timings | 12.367 |
| ast_timings | 10.075 |
| describe_scenes | 9.434 |
| summarize_scenes | 23.578 |
| synthesize_synopsis | 14.576 |
| make_embedding | 1.823 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.441 |
| branch_yolo_total | 7.907 |
| branch_audio_total | 28.962 |
