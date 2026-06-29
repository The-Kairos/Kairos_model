# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:43:01 UTC | Ovp584PUT_c_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.272 | 0.786 | 63.844 | 15.635 | 25.404 | 31.604 | 4.794 |

## 2026-06-25 12:43:01 UTC | Ovp584PUT_c_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ovp584PUT_c_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.272` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.786 |
| save_clips | - |
| sample_frames | 1.180 |
| caption_frames | 50.950 |
| sample_fps | 2.367 |
| detect_object_yolo | 10.319 |
| audio_scan | 15.467 |
| asr_timings | 9.602 |
| ast_timings | 38.767 |
| describe_scenes | 15.635 |
| summarize_scenes | 25.404 |
| synthesize_synopsis | 31.604 |
| make_embedding | 4.794 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 52.136 |
| branch_yolo_total | 12.692 |
| branch_audio_total | 63.844 |
