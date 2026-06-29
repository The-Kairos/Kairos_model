# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:34:21 UTC | ax89Ufgf3ro_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 120.306 | 0.679 | 55.040 | 8.821 | 6.017 | 9.078 | 2.548 |

## 2026-06-26 00:34:21 UTC | ax89Ufgf3ro_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ax89Ufgf3ro_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `120.306` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.679 |
| save_clips | - |
| sample_frames | 0.828 |
| caption_frames | 26.233 |
| sample_fps | 1.982 |
| detect_object_yolo | 7.623 |
| audio_scan | 14.892 |
| asr_timings | 21.297 |
| ast_timings | 18.843 |
| describe_scenes | 8.821 |
| summarize_scenes | 6.017 |
| synthesize_synopsis | 9.078 |
| make_embedding | 2.548 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 27.067 |
| branch_yolo_total | 9.611 |
| branch_audio_total | 55.040 |
