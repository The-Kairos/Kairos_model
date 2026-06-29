# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:26:45 UTC | viPIq7-BdpU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.204 | 0.701 | 36.937 | 6.143 | 6.233 | 8.128 | 2.266 |

## 2026-06-27 02:26:45 UTC | viPIq7-BdpU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/viPIq7-BdpU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.204` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.701 |
| save_clips | - |
| sample_frames | 0.457 |
| caption_frames | 22.569 |
| sample_fps | 1.844 |
| detect_object_yolo | 7.534 |
| audio_scan | 10.726 |
| asr_timings | 10.472 |
| ast_timings | 15.730 |
| describe_scenes | 6.143 |
| summarize_scenes | 6.233 |
| synthesize_synopsis | 8.128 |
| make_embedding | 2.266 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.032 |
| branch_yolo_total | 9.383 |
| branch_audio_total | 36.937 |
