# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 11:15:05 UTC | jZnB2yjitMs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 260.327 | 0.805 | 73.024 | 30.929 | 47.848 | 19.248 | 5.795 |

## 2026-06-26 11:15:05 UTC | jZnB2yjitMs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jZnB2yjitMs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `260.327` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.805 |
| save_clips | - |
| sample_frames | 1.526 |
| caption_frames | 64.927 |
| sample_fps | 2.572 |
| detect_object_yolo | 12.198 |
| audio_scan | 15.101 |
| asr_timings | 11.003 |
| ast_timings | 46.912 |
| describe_scenes | 30.929 |
| summarize_scenes | 47.848 |
| synthesize_synopsis | 19.248 |
| make_embedding | 5.795 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 66.459 |
| branch_yolo_total | 14.776 |
| branch_audio_total | 73.024 |
