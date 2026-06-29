# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:29:54 UTC | fVJmyaS_w4M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.829 | 0.636 | 48.244 | 4.323 | 8.973 | 31.334 | 2.072 |

## 2026-06-26 04:29:54 UTC | fVJmyaS_w4M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fVJmyaS_w4M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.829` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.636 |
| save_clips | - |
| sample_frames | 0.362 |
| caption_frames | 18.944 |
| sample_fps | 1.741 |
| detect_object_yolo | 6.807 |
| audio_scan | 15.084 |
| asr_timings | 20.835 |
| ast_timings | 12.316 |
| describe_scenes | 4.323 |
| summarize_scenes | 8.973 |
| synthesize_synopsis | 31.334 |
| make_embedding | 2.072 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.311 |
| branch_yolo_total | 8.554 |
| branch_audio_total | 48.244 |
