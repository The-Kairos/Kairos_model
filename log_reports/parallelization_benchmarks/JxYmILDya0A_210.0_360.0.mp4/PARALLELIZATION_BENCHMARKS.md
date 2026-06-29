# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:25:26 UTC | JxYmILDya0A_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.820 | 0.829 | 57.164 | 21.335 | 11.335 | 13.735 | 3.639 |

## 2026-06-25 05:25:26 UTC | JxYmILDya0A_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/JxYmILDya0A_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.820` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.829 |
| save_clips | - |
| sample_frames | 1.475 |
| caption_frames | 40.578 |
| sample_fps | 2.400 |
| detect_object_yolo | 8.935 |
| audio_scan | 12.886 |
| asr_timings | 14.291 |
| ast_timings | 29.978 |
| describe_scenes | 21.335 |
| summarize_scenes | 11.335 |
| synthesize_synopsis | 13.735 |
| make_embedding | 3.639 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.059 |
| branch_yolo_total | 11.340 |
| branch_audio_total | 57.164 |
