# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:44:52 UTC | boJssjt0HGk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 229.006 | 0.684 | 70.012 | 17.755 | 43.906 | 12.002 | 5.850 |

## 2026-06-26 01:44:52 UTC | boJssjt0HGk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/boJssjt0HGk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `229.006` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.684 |
| save_clips | - |
| sample_frames | 1.493 |
| caption_frames | 61.350 |
| sample_fps | 2.382 |
| detect_object_yolo | 12.080 |
| audio_scan | 13.124 |
| asr_timings | 8.984 |
| ast_timings | 47.895 |
| describe_scenes | 17.755 |
| summarize_scenes | 43.906 |
| synthesize_synopsis | 12.002 |
| make_embedding | 5.850 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 62.849 |
| branch_yolo_total | 14.468 |
| branch_audio_total | 70.012 |
