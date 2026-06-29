# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:40:40 UTC | 5Ib6GnYyw-o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 155.570 | 0.860 | 52.727 | 11.518 | 8.371 | 22.170 | 3.693 |

## 2026-06-24 11:40:40 UTC | 5Ib6GnYyw-o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5Ib6GnYyw-o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `155.570` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.860 |
| save_clips | - |
| sample_frames | 1.455 |
| caption_frames | 41.958 |
| sample_fps | 2.350 |
| detect_object_yolo | 9.080 |
| audio_scan | 11.760 |
| asr_timings | 11.820 |
| ast_timings | 29.138 |
| describe_scenes | 11.518 |
| summarize_scenes | 8.371 |
| synthesize_synopsis | 22.170 |
| make_embedding | 3.693 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.420 |
| branch_yolo_total | 11.436 |
| branch_audio_total | 52.727 |
