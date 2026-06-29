# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:14:00 UTC | 6MTL7aBxgR0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.344 | 0.819 | 42.799 | 21.680 | 22.120 | 17.895 | 3.880 |

## 2026-06-24 12:14:00 UTC | 6MTL7aBxgR0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6MTL7aBxgR0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.344` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 1.246 |
| caption_frames | 41.547 |
| sample_fps | 2.324 |
| detect_object_yolo | 8.765 |
| audio_scan | 3.864 |
| asr_timings | 0.000 |
| ast_timings | 32.789 |
| describe_scenes | 21.680 |
| summarize_scenes | 22.120 |
| synthesize_synopsis | 17.895 |
| make_embedding | 3.880 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.799 |
| branch_yolo_total | 11.094 |
| branch_audio_total | 36.662 |
