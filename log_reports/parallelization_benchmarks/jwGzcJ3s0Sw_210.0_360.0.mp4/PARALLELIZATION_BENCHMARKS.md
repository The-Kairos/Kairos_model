# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:31:47 UTC | jwGzcJ3s0Sw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 101.451 | 0.810 | 24.656 | 9.482 | 25.100 | 22.583 | 1.372 |

## 2026-06-26 12:31:47 UTC | jwGzcJ3s0Sw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jwGzcJ3s0Sw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `101.451` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.810 |
| save_clips | - |
| sample_frames | 0.102 |
| caption_frames | 8.733 |
| sample_fps | 1.777 |
| detect_object_yolo | 5.440 |
| audio_scan | 11.927 |
| asr_timings | 8.081 |
| ast_timings | 4.639 |
| describe_scenes | 9.482 |
| summarize_scenes | 25.100 |
| synthesize_synopsis | 22.583 |
| make_embedding | 1.372 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.841 |
| branch_yolo_total | 7.224 |
| branch_audio_total | 24.656 |
