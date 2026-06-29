# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:34:13 UTC | GErRlbPkMmQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.990 | 1.563 | 39.245 | 5.276 | 13.659 | 21.191 | 2.016 |

## 2026-06-25 01:34:13 UTC | GErRlbPkMmQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GErRlbPkMmQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.990` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.563 |
| save_clips | - |
| sample_frames | 0.500 |
| caption_frames | 17.655 |
| sample_fps | 1.952 |
| detect_object_yolo | 6.469 |
| audio_scan | 16.103 |
| asr_timings | 10.031 |
| ast_timings | 13.103 |
| describe_scenes | 5.276 |
| summarize_scenes | 13.659 |
| synthesize_synopsis | 21.191 |
| make_embedding | 2.016 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.161 |
| branch_yolo_total | 8.426 |
| branch_audio_total | 39.245 |
