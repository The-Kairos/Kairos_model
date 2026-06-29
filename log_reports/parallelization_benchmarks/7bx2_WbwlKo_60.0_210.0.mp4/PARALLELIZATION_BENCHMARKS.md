# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:28:04 UTC | 7bx2_WbwlKo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.939 | 0.833 | 56.074 | 25.555 | 10.601 | 28.858 | 3.896 |

## 2026-06-24 14:28:04 UTC | 7bx2_WbwlKo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7bx2_WbwlKo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.939` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.833 |
| save_clips | - |
| sample_frames | 1.149 |
| caption_frames | 42.204 |
| sample_fps | 2.292 |
| detect_object_yolo | 9.079 |
| audio_scan | 13.812 |
| asr_timings | 9.351 |
| ast_timings | 32.903 |
| describe_scenes | 25.555 |
| summarize_scenes | 10.601 |
| synthesize_synopsis | 28.858 |
| make_embedding | 3.896 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.359 |
| branch_yolo_total | 11.377 |
| branch_audio_total | 56.074 |
