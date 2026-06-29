# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 17:10:47 UTC | oN4XuLnG7Gw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.415 | 0.854 | 56.667 | 11.741 | 8.820 | 8.948 | 4.166 |

## 2026-06-27 17:10:47 UTC | oN4XuLnG7Gw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/oN4XuLnG7Gw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.415` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.854 |
| save_clips | - |
| sample_frames | 1.492 |
| caption_frames | 45.277 |
| sample_fps | 2.504 |
| detect_object_yolo | 9.544 |
| audio_scan | 12.735 |
| asr_timings | 8.263 |
| ast_timings | 35.661 |
| describe_scenes | 11.741 |
| summarize_scenes | 8.820 |
| synthesize_synopsis | 8.948 |
| make_embedding | 4.166 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.775 |
| branch_yolo_total | 12.053 |
| branch_audio_total | 56.667 |
