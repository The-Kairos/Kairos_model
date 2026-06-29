# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:37:57 UTC | ER7WWu1fPNI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.613 | 0.800 | 56.291 | 14.785 | 7.666 | 5.098 | 3.727 |

## 2026-06-24 23:37:57 UTC | ER7WWu1fPNI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ER7WWu1fPNI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.613` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.401 |
| caption_frames | 37.103 |
| sample_fps | 2.351 |
| detect_object_yolo | 8.961 |
| audio_scan | 7.594 |
| asr_timings | 18.396 |
| ast_timings | 30.293 |
| describe_scenes | 14.785 |
| summarize_scenes | 7.666 |
| synthesize_synopsis | 5.098 |
| make_embedding | 3.727 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.510 |
| branch_yolo_total | 11.318 |
| branch_audio_total | 56.291 |
