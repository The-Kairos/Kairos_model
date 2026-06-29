# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:00:53 UTC | QueGIYya64M_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 185.019 | 0.760 | 76.387 | 17.021 | 19.164 | 18.829 | 3.386 |

## 2026-06-25 16:00:53 UTC | QueGIYya64M_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/QueGIYya64M_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `185.019` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 0.886 |
| caption_frames | 36.550 |
| sample_fps | 2.173 |
| detect_object_yolo | 8.438 |
| audio_scan | 11.167 |
| asr_timings | 38.225 |
| ast_timings | 26.987 |
| describe_scenes | 17.021 |
| summarize_scenes | 19.164 |
| synthesize_synopsis | 18.829 |
| make_embedding | 3.386 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.443 |
| branch_yolo_total | 10.617 |
| branch_audio_total | 76.387 |
