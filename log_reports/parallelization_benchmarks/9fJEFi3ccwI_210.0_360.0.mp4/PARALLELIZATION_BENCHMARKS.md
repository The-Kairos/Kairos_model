# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:23:50 UTC | 9fJEFi3ccwI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 210.008 | 0.953 | 68.706 | 21.171 | 26.129 | 12.347 | 5.452 |

## 2026-06-24 18:23:50 UTC | 9fJEFi3ccwI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9fJEFi3ccwI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `210.008` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.953 |
| save_clips | - |
| sample_frames | 2.039 |
| caption_frames | 57.802 |
| sample_fps | 2.916 |
| detect_object_yolo | 11.074 |
| audio_scan | 14.941 |
| asr_timings | 9.012 |
| ast_timings | 44.744 |
| describe_scenes | 21.171 |
| summarize_scenes | 26.129 |
| synthesize_synopsis | 12.347 |
| make_embedding | 5.452 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.847 |
| branch_yolo_total | 13.996 |
| branch_audio_total | 68.706 |
