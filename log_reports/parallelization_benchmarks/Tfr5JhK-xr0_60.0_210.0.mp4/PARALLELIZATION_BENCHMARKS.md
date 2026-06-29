# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:57:21 UTC | Tfr5JhK-xr0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 235.869 | 0.800 | 78.949 | 19.768 | 17.413 | 17.588 | 6.773 |

## 2026-06-25 17:57:21 UTC | Tfr5JhK-xr0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Tfr5JhK-xr0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `235.869` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.770 |
| caption_frames | 77.173 |
| sample_fps | 2.677 |
| detect_object_yolo | 13.668 |
| audio_scan | 6.512 |
| asr_timings | 7.821 |
| ast_timings | 62.476 |
| describe_scenes | 19.768 |
| summarize_scenes | 17.413 |
| synthesize_synopsis | 17.588 |
| make_embedding | 6.773 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 78.949 |
| branch_yolo_total | 16.350 |
| branch_audio_total | 76.817 |
