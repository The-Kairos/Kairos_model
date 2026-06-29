# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 17:05:15 UTC | oN4XuLnG7Gw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.450 | 0.893 | 56.911 | 10.838 | 12.273 | 10.509 | 4.635 |

## 2026-06-27 17:05:15 UTC | oN4XuLnG7Gw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/oN4XuLnG7Gw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.450` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.893 |
| save_clips | - |
| sample_frames | 1.425 |
| caption_frames | 47.585 |
| sample_fps | 2.433 |
| detect_object_yolo | 9.567 |
| audio_scan | 12.686 |
| asr_timings | 9.326 |
| ast_timings | 34.891 |
| describe_scenes | 10.838 |
| summarize_scenes | 12.273 |
| synthesize_synopsis | 10.509 |
| make_embedding | 4.635 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.017 |
| branch_yolo_total | 12.006 |
| branch_audio_total | 56.911 |
