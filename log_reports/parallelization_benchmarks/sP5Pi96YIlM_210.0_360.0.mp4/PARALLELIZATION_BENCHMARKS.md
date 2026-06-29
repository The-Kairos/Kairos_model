# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:46:08 UTC | sP5Pi96YIlM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 237.018 | 0.781 | 79.874 | 31.610 | 14.964 | 13.913 | 6.584 |

## 2026-06-26 19:46:08 UTC | sP5Pi96YIlM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sP5Pi96YIlM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `237.018` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.660 |
| caption_frames | 70.590 |
| sample_fps | 2.528 |
| detect_object_yolo | 13.073 |
| audio_scan | 13.200 |
| asr_timings | 12.370 |
| ast_timings | 54.296 |
| describe_scenes | 31.610 |
| summarize_scenes | 14.964 |
| synthesize_synopsis | 13.913 |
| make_embedding | 6.584 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 72.255 |
| branch_yolo_total | 15.607 |
| branch_audio_total | 79.874 |
