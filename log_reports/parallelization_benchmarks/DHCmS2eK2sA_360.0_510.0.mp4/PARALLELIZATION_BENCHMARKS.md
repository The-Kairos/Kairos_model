# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:22:41 UTC | DHCmS2eK2sA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 132.024 | 0.746 | 47.477 | 8.600 | 11.230 | 15.145 | 2.992 |

## 2026-06-24 22:22:41 UTC | DHCmS2eK2sA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DHCmS2eK2sA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `132.024` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.746 |
| save_clips | - |
| sample_frames | 0.699 |
| caption_frames | 33.629 |
| sample_fps | 2.007 |
| detect_object_yolo | 8.083 |
| audio_scan | 12.725 |
| asr_timings | 10.334 |
| ast_timings | 24.410 |
| describe_scenes | 8.600 |
| summarize_scenes | 11.230 |
| synthesize_synopsis | 15.145 |
| make_embedding | 2.992 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.333 |
| branch_yolo_total | 10.095 |
| branch_audio_total | 47.477 |
