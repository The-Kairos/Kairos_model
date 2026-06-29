# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:08:32 UTC | Z2eyocRl2mo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.115 | 0.624 | 57.471 | 15.080 | 11.977 | 7.825 | 4.473 |

## 2026-06-25 22:08:32 UTC | Z2eyocRl2mo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Z2eyocRl2mo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.115` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.624 |
| save_clips | - |
| sample_frames | 1.135 |
| caption_frames | 48.885 |
| sample_fps | 2.208 |
| detect_object_yolo | 10.012 |
| audio_scan | 6.458 |
| asr_timings | 12.799 |
| ast_timings | 38.205 |
| describe_scenes | 15.080 |
| summarize_scenes | 11.977 |
| synthesize_synopsis | 7.825 |
| make_embedding | 4.473 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.026 |
| branch_yolo_total | 12.226 |
| branch_audio_total | 57.471 |
