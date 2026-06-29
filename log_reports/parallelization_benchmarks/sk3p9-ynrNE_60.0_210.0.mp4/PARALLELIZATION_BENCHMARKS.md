# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 21:56:52 UTC | sk3p9-ynrNE_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 1891.962 | 0.779 | 1797.959 | 13.526 | 10.373 | 10.144 | 3.563 |

## 2026-06-26 21:56:52 UTC | sk3p9-ynrNE_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sk3p9-ynrNE_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1891.962` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.397 |
| caption_frames | 41.478 |
| sample_fps | 2.316 |
| detect_object_yolo | 8.990 |
| audio_scan | 12.759 |
| asr_timings | 1754.510 |
| ast_timings | 30.681 |
| describe_scenes | 13.526 |
| summarize_scenes | 10.373 |
| synthesize_synopsis | 10.144 |
| make_embedding | 3.563 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.881 |
| branch_yolo_total | 11.312 |
| branch_audio_total | 1797.959 |
