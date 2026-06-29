# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:47:13 UTC | XXDq63KUBNo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.071 | 0.819 | 48.542 | 15.564 | 17.198 | 14.003 | 3.088 |

## 2026-06-25 21:47:13 UTC | XXDq63KUBNo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/XXDq63KUBNo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.071` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 0.839 |
| caption_frames | 32.775 |
| sample_fps | 2.144 |
| detect_object_yolo | 7.703 |
| audio_scan | 15.063 |
| asr_timings | 8.597 |
| ast_timings | 24.874 |
| describe_scenes | 15.564 |
| summarize_scenes | 17.198 |
| synthesize_synopsis | 14.003 |
| make_embedding | 3.088 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.619 |
| branch_yolo_total | 9.853 |
| branch_audio_total | 48.542 |
