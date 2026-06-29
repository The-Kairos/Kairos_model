# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 21:36:48 UTC | CyAyEewgdEc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 1910.919 | 0.777 | 1807.815 | 12.695 | 12.554 | 13.265 | 3.848 |

## 2026-06-24 21:36:48 UTC | CyAyEewgdEc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CyAyEewgdEc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `1910.919` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.777 |
| save_clips | - |
| sample_frames | 0.964 |
| caption_frames | 45.226 |
| sample_fps | 2.285 |
| detect_object_yolo | 10.081 |
| audio_scan | 6.532 |
| asr_timings | 1768.367 |
| ast_timings | 32.907 |
| describe_scenes | 12.695 |
| summarize_scenes | 12.554 |
| synthesize_synopsis | 13.265 |
| make_embedding | 3.848 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.196 |
| branch_yolo_total | 12.372 |
| branch_audio_total | 1807.815 |
