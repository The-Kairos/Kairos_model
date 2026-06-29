# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:21:33 UTC | lxf17LCvWoM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 226.272 | 0.789 | 71.945 | 25.454 | 16.923 | 27.863 | 5.723 |

## 2026-06-26 17:21:33 UTC | lxf17LCvWoM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lxf17LCvWoM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `226.272` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.511 |
| caption_frames | 59.824 |
| sample_fps | 2.565 |
| detect_object_yolo | 12.255 |
| audio_scan | 10.727 |
| asr_timings | 13.746 |
| ast_timings | 47.458 |
| describe_scenes | 25.454 |
| summarize_scenes | 16.923 |
| synthesize_synopsis | 27.863 |
| make_embedding | 5.723 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.341 |
| branch_yolo_total | 14.825 |
| branch_audio_total | 71.945 |
