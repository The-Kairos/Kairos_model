# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-21 22:11:57 UTC | 3UPP_WRL86c_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 125.572 | 0.839 | 62.152 | 5.602 | 5.610 | 6.737 | 2.849 |

## 2026-06-21 22:11:57 UTC | 3UPP_WRL86c_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/3UPP_WRL86c_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `125.572` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.839 |
| save_clips | - |
| sample_frames | 1.026 |
| caption_frames | 29.627 |
| sample_fps | 2.169 |
| detect_object_yolo | 7.574 |
| audio_scan | 9.683 |
| asr_timings | 30.942 |
| ast_timings | 21.518 |
| describe_scenes | 5.602 |
| summarize_scenes | 5.610 |
| synthesize_synopsis | 6.737 |
| make_embedding | 2.849 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.659 |
| branch_yolo_total | 9.748 |
| branch_audio_total | 62.152 |
