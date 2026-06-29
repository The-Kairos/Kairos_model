# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:40:47 UTC | kM_8DQ-iJcU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 139.033 | 0.661 | 47.982 | 21.396 | 13.259 | 17.604 | 2.504 |

## 2026-06-26 13:40:47 UTC | kM_8DQ-iJcU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kM_8DQ-iJcU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `139.033` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.661 |
| save_clips | - |
| sample_frames | 0.684 |
| caption_frames | 24.711 |
| sample_fps | 1.905 |
| detect_object_yolo | 6.915 |
| audio_scan | 5.454 |
| asr_timings | 24.246 |
| ast_timings | 18.274 |
| describe_scenes | 21.396 |
| summarize_scenes | 13.259 |
| synthesize_synopsis | 17.604 |
| make_embedding | 2.504 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 25.401 |
| branch_yolo_total | 8.826 |
| branch_audio_total | 47.982 |
