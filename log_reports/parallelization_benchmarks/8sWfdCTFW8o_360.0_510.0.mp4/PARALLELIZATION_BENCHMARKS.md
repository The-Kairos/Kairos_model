# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:15:40 UTC | 8sWfdCTFW8o_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.029 | 0.708 | 54.854 | 11.114 | 9.591 | 20.640 | 3.879 |

## 2026-06-24 17:15:40 UTC | 8sWfdCTFW8o_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8sWfdCTFW8o_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.029` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.708 |
| save_clips | - |
| sample_frames | 1.127 |
| caption_frames | 43.747 |
| sample_fps | 2.189 |
| detect_object_yolo | 9.701 |
| audio_scan | 8.638 |
| asr_timings | 13.625 |
| ast_timings | 32.582 |
| describe_scenes | 11.114 |
| summarize_scenes | 9.591 |
| synthesize_synopsis | 20.640 |
| make_embedding | 3.879 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.880 |
| branch_yolo_total | 11.896 |
| branch_audio_total | 54.854 |
