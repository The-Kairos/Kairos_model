# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:32:29 UTC | b_gbAILvCQo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.682 | 0.647 | 73.854 | 23.512 | 11.232 | 10.031 | 5.527 |

## 2026-06-26 01:32:29 UTC | b_gbAILvCQo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/b_gbAILvCQo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.682` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.647 |
| save_clips | - |
| sample_frames | 1.614 |
| caption_frames | 57.786 |
| sample_fps | 2.417 |
| detect_object_yolo | 11.636 |
| audio_scan | 11.877 |
| asr_timings | 17.370 |
| ast_timings | 44.599 |
| describe_scenes | 23.512 |
| summarize_scenes | 11.232 |
| synthesize_synopsis | 10.031 |
| make_embedding | 5.527 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.406 |
| branch_yolo_total | 14.059 |
| branch_audio_total | 73.854 |
