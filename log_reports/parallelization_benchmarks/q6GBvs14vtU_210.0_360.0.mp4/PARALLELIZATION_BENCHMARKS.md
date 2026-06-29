# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:31:41 UTC | q6GBvs14vtU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 143.302 | 0.671 | 58.403 | 8.321 | 6.964 | 6.990 | 3.850 |

## 2026-06-28 08:31:41 UTC | q6GBvs14vtU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q6GBvs14vtU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `143.302` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.671 |
| save_clips | - |
| sample_frames | 1.318 |
| caption_frames | 43.710 |
| sample_fps | 2.276 |
| detect_object_yolo | 9.406 |
| audio_scan | 14.919 |
| asr_timings | 10.749 |
| ast_timings | 32.726 |
| describe_scenes | 8.321 |
| summarize_scenes | 6.964 |
| synthesize_synopsis | 6.990 |
| make_embedding | 3.850 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.034 |
| branch_yolo_total | 11.687 |
| branch_audio_total | 58.403 |
