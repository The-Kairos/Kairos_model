# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:29:19 UTC | RLh2fCh9SpA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.744 | 0.815 | 60.932 | 15.406 | 20.997 | 20.794 | 2.829 |

## 2026-06-25 16:29:19 UTC | RLh2fCh9SpA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/RLh2fCh9SpA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.744` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.815 |
| save_clips | - |
| sample_frames | 1.134 |
| caption_frames | 30.301 |
| sample_fps | 2.264 |
| detect_object_yolo | 7.840 |
| audio_scan | 15.617 |
| asr_timings | 23.744 |
| ast_timings | 21.563 |
| describe_scenes | 15.406 |
| summarize_scenes | 20.997 |
| synthesize_synopsis | 20.794 |
| make_embedding | 2.829 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.440 |
| branch_yolo_total | 10.110 |
| branch_audio_total | 60.932 |
