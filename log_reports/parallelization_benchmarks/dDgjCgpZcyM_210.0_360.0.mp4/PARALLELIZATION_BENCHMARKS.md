# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:46:11 UTC | dDgjCgpZcyM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 113.856 | 0.635 | 38.674 | 10.807 | 9.950 | 15.582 | 2.497 |

## 2026-06-26 02:46:11 UTC | dDgjCgpZcyM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/dDgjCgpZcyM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `113.856` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.635 |
| save_clips | - |
| sample_frames | 0.598 |
| caption_frames | 24.385 |
| sample_fps | 1.845 |
| detect_object_yolo | 7.476 |
| audio_scan | 9.766 |
| asr_timings | 10.242 |
| ast_timings | 18.657 |
| describe_scenes | 10.807 |
| summarize_scenes | 9.950 |
| synthesize_synopsis | 15.582 |
| make_embedding | 2.497 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.989 |
| branch_yolo_total | 9.326 |
| branch_audio_total | 38.674 |
