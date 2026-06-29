# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:42:51 UTC | AIYfAAX7XL0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 122.373 | 0.656 | 45.131 | 10.577 | 8.067 | 15.354 | 2.804 |

## 2026-06-24 18:42:51 UTC | AIYfAAX7XL0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AIYfAAX7XL0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `122.373` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 0.786 |
| caption_frames | 28.194 |
| sample_fps | 1.930 |
| detect_object_yolo | 7.478 |
| audio_scan | 13.923 |
| asr_timings | 9.821 |
| ast_timings | 21.378 |
| describe_scenes | 10.577 |
| summarize_scenes | 8.067 |
| synthesize_synopsis | 15.354 |
| make_embedding | 2.804 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.986 |
| branch_yolo_total | 9.414 |
| branch_audio_total | 45.131 |
