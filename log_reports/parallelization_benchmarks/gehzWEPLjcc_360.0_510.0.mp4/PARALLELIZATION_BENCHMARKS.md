# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:21:27 UTC | gehzWEPLjcc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 89.404 | 0.727 | 31.103 | 5.178 | 8.175 | 15.891 | 2.088 |

## 2026-06-26 05:21:27 UTC | gehzWEPLjcc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/gehzWEPLjcc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `89.404` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.727 |
| save_clips | - |
| sample_frames | 0.389 |
| caption_frames | 15.681 |
| sample_fps | 1.761 |
| detect_object_yolo | 6.961 |
| audio_scan | 11.906 |
| asr_timings | 8.918 |
| ast_timings | 10.270 |
| describe_scenes | 5.178 |
| summarize_scenes | 8.175 |
| synthesize_synopsis | 15.891 |
| make_embedding | 2.088 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 16.076 |
| branch_yolo_total | 8.728 |
| branch_audio_total | 31.103 |
