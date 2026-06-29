# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:10:38 UTC | wR-XEd-TBJk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.791 | 0.757 | 47.339 | 7.368 | 10.209 | 10.858 | 3.112 |

## 2026-06-27 03:10:38 UTC | wR-XEd-TBJk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/wR-XEd-TBJk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.791` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.757 |
| save_clips | - |
| sample_frames | 0.693 |
| caption_frames | 32.201 |
| sample_fps | 2.070 |
| detect_object_yolo | 7.780 |
| audio_scan | 14.072 |
| asr_timings | 8.876 |
| ast_timings | 24.382 |
| describe_scenes | 7.368 |
| summarize_scenes | 10.209 |
| synthesize_synopsis | 10.858 |
| make_embedding | 3.112 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.900 |
| branch_yolo_total | 9.856 |
| branch_audio_total | 47.339 |
