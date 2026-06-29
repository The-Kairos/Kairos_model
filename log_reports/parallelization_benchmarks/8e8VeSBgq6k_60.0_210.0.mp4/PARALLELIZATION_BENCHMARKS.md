# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 16:54:55 UTC | 8e8VeSBgq6k_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 239.244 | 0.809 | 80.860 | 23.477 | 17.324 | 16.282 | 6.740 |

## 2026-06-24 16:54:55 UTC | 8e8VeSBgq6k_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8e8VeSBgq6k_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `239.244` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.809 |
| save_clips | - |
| sample_frames | 1.555 |
| caption_frames | 73.937 |
| sample_fps | 2.707 |
| detect_object_yolo | 14.084 |
| audio_scan | 15.054 |
| asr_timings | 12.124 |
| ast_timings | 53.674 |
| describe_scenes | 23.477 |
| summarize_scenes | 17.324 |
| synthesize_synopsis | 16.282 |
| make_embedding | 6.740 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 75.499 |
| branch_yolo_total | 16.797 |
| branch_audio_total | 80.860 |
