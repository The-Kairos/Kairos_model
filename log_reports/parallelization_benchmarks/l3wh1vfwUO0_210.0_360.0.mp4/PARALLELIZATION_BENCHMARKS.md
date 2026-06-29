# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:30:21 UTC | l3wh1vfwUO0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 218.637 | 0.806 | 81.891 | 26.368 | 21.029 | 23.489 | 4.150 |

## 2026-06-26 14:30:21 UTC | l3wh1vfwUO0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/l3wh1vfwUO0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `218.637` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.806 |
| save_clips | - |
| sample_frames | 1.640 |
| caption_frames | 46.263 |
| sample_fps | 2.423 |
| detect_object_yolo | 9.152 |
| audio_scan | 11.976 |
| asr_timings | 33.283 |
| ast_timings | 36.623 |
| describe_scenes | 26.368 |
| summarize_scenes | 21.029 |
| synthesize_synopsis | 23.489 |
| make_embedding | 4.150 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.909 |
| branch_yolo_total | 11.581 |
| branch_audio_total | 81.891 |
