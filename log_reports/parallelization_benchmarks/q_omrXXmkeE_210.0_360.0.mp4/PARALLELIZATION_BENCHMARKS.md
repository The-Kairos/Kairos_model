# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 08:48:39 UTC | q_omrXXmkeE_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 169.693 | 0.792 | 62.353 | 15.973 | 10.915 | 7.785 | 4.491 |

## 2026-06-28 08:48:39 UTC | q_omrXXmkeE_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/q_omrXXmkeE_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `169.693` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.639 |
| caption_frames | 51.436 |
| sample_fps | 2.478 |
| detect_object_yolo | 10.394 |
| audio_scan | 14.941 |
| asr_timings | 9.155 |
| ast_timings | 38.248 |
| describe_scenes | 15.973 |
| summarize_scenes | 10.915 |
| synthesize_synopsis | 7.785 |
| make_embedding | 4.491 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.081 |
| branch_yolo_total | 12.878 |
| branch_audio_total | 62.353 |
