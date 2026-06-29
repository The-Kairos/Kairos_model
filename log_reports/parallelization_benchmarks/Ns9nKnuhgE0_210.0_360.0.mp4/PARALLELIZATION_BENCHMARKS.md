# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:53:04 UTC | Ns9nKnuhgE0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 198.599 | 0.868 | 62.199 | 25.896 | 16.624 | 20.800 | 4.799 |

## 2026-06-25 10:53:04 UTC | Ns9nKnuhgE0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ns9nKnuhgE0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `198.599` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.868 |
| save_clips | - |
| sample_frames | 1.463 |
| caption_frames | 51.707 |
| sample_fps | 2.503 |
| detect_object_yolo | 10.332 |
| audio_scan | 13.861 |
| asr_timings | 9.608 |
| ast_timings | 38.721 |
| describe_scenes | 25.896 |
| summarize_scenes | 16.624 |
| synthesize_synopsis | 20.800 |
| make_embedding | 4.799 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.175 |
| branch_yolo_total | 12.841 |
| branch_audio_total | 62.199 |
