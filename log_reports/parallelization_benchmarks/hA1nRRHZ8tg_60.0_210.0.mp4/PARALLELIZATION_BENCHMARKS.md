# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:09:09 UTC | hA1nRRHZ8tg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.620 | 0.791 | 64.404 | 21.897 | 14.702 | 24.151 | 5.052 |

## 2026-06-26 06:09:09 UTC | hA1nRRHZ8tg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hA1nRRHZ8tg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.620` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.181 |
| caption_frames | 55.761 |
| sample_fps | 2.399 |
| detect_object_yolo | 10.855 |
| audio_scan | 14.966 |
| asr_timings | 6.965 |
| ast_timings | 42.465 |
| describe_scenes | 21.897 |
| summarize_scenes | 14.702 |
| synthesize_synopsis | 24.151 |
| make_embedding | 5.052 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.947 |
| branch_yolo_total | 13.260 |
| branch_audio_total | 64.404 |
