# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 17:01:09 UTC | S8ZUWshc1C4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 200.080 | 0.645 | 65.759 | 17.109 | 26.116 | 15.098 | 5.423 |

## 2026-06-25 17:01:09 UTC | S8ZUWshc1C4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/S8ZUWshc1C4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `200.080` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.645 |
| save_clips | - |
| sample_frames | 1.476 |
| caption_frames | 53.596 |
| sample_fps | 2.401 |
| detect_object_yolo | 11.032 |
| audio_scan | 15.754 |
| asr_timings | 10.050 |
| ast_timings | 39.947 |
| describe_scenes | 17.109 |
| summarize_scenes | 26.116 |
| synthesize_synopsis | 15.098 |
| make_embedding | 5.423 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.078 |
| branch_yolo_total | 13.439 |
| branch_audio_total | 65.759 |
