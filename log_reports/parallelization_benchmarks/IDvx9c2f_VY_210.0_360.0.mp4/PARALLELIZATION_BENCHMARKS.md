# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:17:15 UTC | IDvx9c2f_VY_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 194.689 | 0.721 | 69.805 | 16.308 | 11.825 | 13.376 | 5.762 |

## 2026-06-25 04:17:15 UTC | IDvx9c2f_VY_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IDvx9c2f_VY_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `194.689` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.721 |
| save_clips | - |
| sample_frames | 1.820 |
| caption_frames | 59.584 |
| sample_fps | 2.639 |
| detect_object_yolo | 11.417 |
| audio_scan | 14.852 |
| asr_timings | 9.236 |
| ast_timings | 45.709 |
| describe_scenes | 16.308 |
| summarize_scenes | 11.825 |
| synthesize_synopsis | 13.376 |
| make_embedding | 5.762 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.410 |
| branch_yolo_total | 14.062 |
| branch_audio_total | 69.805 |
