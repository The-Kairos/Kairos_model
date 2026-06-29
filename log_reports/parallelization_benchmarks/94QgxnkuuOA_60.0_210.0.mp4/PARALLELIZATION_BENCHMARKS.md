# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:35:03 UTC | 94QgxnkuuOA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 86.107 | 1.536 | 28.238 | 6.655 | 7.041 | 17.334 | 2.061 |

## 2026-06-24 17:35:03 UTC | 94QgxnkuuOA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/94QgxnkuuOA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `86.107` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.536 |
| save_clips | - |
| sample_frames | 0.330 |
| caption_frames | 14.363 |
| sample_fps | 1.866 |
| detect_object_yolo | 5.299 |
| audio_scan | 8.571 |
| asr_timings | 9.841 |
| ast_timings | 9.818 |
| describe_scenes | 6.655 |
| summarize_scenes | 7.041 |
| synthesize_synopsis | 17.334 |
| make_embedding | 2.061 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.699 |
| branch_yolo_total | 7.171 |
| branch_audio_total | 28.238 |
