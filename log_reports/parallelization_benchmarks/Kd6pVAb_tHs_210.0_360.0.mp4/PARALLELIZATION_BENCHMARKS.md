# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:44:20 UTC | Kd6pVAb_tHs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.853 | 0.852 | 33.988 | 13.813 | 9.869 | 9.967 | 2.804 |

## 2026-06-25 06:44:20 UTC | Kd6pVAb_tHs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Kd6pVAb_tHs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.853` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.852 |
| save_clips | - |
| sample_frames | 1.288 |
| caption_frames | 32.694 |
| sample_fps | 2.243 |
| detect_object_yolo | 7.567 |
| audio_scan | 3.834 |
| asr_timings | 0.000 |
| ast_timings | 21.525 |
| describe_scenes | 13.813 |
| summarize_scenes | 9.869 |
| synthesize_synopsis | 9.967 |
| make_embedding | 2.804 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.988 |
| branch_yolo_total | 9.816 |
| branch_audio_total | 25.368 |
