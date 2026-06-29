# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:33:23 UTC | yvx5SmeFdjM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 205.128 | 0.770 | 99.703 | 16.448 | 14.880 | 9.748 | 4.092 |

## 2026-06-27 05:33:23 UTC | yvx5SmeFdjM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yvx5SmeFdjM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `205.128` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.770 |
| save_clips | - |
| sample_frames | 1.188 |
| caption_frames | 44.967 |
| sample_fps | 2.316 |
| detect_object_yolo | 9.607 |
| audio_scan | 14.999 |
| asr_timings | 49.886 |
| ast_timings | 34.810 |
| describe_scenes | 16.448 |
| summarize_scenes | 14.880 |
| synthesize_synopsis | 9.748 |
| make_embedding | 4.092 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.161 |
| branch_yolo_total | 11.928 |
| branch_audio_total | 99.703 |
