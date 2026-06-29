# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:47:13 UTC | TBMt3agq6_w_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 168.698 | 0.773 | 56.950 | 18.608 | 11.333 | 17.363 | 4.200 |

## 2026-06-25 17:47:13 UTC | TBMt3agq6_w_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TBMt3agq6_w_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `168.698` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 1.167 |
| caption_frames | 44.630 |
| sample_fps | 2.317 |
| detect_object_yolo | 9.911 |
| audio_scan | 9.793 |
| asr_timings | 10.739 |
| ast_timings | 36.409 |
| describe_scenes | 18.608 |
| summarize_scenes | 11.333 |
| synthesize_synopsis | 17.363 |
| make_embedding | 4.200 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.804 |
| branch_yolo_total | 12.233 |
| branch_audio_total | 56.950 |
