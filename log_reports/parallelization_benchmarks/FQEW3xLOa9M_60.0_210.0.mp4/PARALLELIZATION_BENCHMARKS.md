# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:17:57 UTC | FQEW3xLOa9M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 92.671 | 0.774 | 33.645 | 6.848 | 5.374 | 10.119 | 2.260 |

## 2026-06-25 00:17:57 UTC | FQEW3xLOa9M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/FQEW3xLOa9M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `92.671` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 0.572 |
| caption_frames | 22.886 |
| sample_fps | 1.959 |
| detect_object_yolo | 6.816 |
| audio_scan | 6.542 |
| asr_timings | 11.442 |
| ast_timings | 15.652 |
| describe_scenes | 6.848 |
| summarize_scenes | 5.374 |
| synthesize_synopsis | 10.119 |
| make_embedding | 2.260 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.463 |
| branch_yolo_total | 8.781 |
| branch_audio_total | 33.645 |
