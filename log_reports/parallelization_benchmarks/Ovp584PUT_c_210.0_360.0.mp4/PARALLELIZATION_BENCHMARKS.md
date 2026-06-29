# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:36:41 UTC | Ovp584PUT_c_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.545 | 0.796 | 54.274 | 21.853 | 18.851 | 25.126 | 3.597 |

## 2026-06-25 12:36:41 UTC | Ovp584PUT_c_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ovp584PUT_c_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.545` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.152 |
| caption_frames | 40.056 |
| sample_fps | 2.316 |
| detect_object_yolo | 9.130 |
| audio_scan | 13.273 |
| asr_timings | 10.978 |
| ast_timings | 30.014 |
| describe_scenes | 21.853 |
| summarize_scenes | 18.851 |
| synthesize_synopsis | 25.126 |
| make_embedding | 3.597 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.214 |
| branch_yolo_total | 11.452 |
| branch_audio_total | 54.274 |
