# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:14:55 UTC | 9erWlhsParM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.946 | 0.633 | 50.304 | 12.839 | 8.247 | 21.100 | 3.080 |

## 2026-06-24 18:14:55 UTC | 9erWlhsParM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/9erWlhsParM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.946` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.633 |
| save_clips | - |
| sample_frames | 0.765 |
| caption_frames | 32.941 |
| sample_fps | 1.919 |
| detect_object_yolo | 7.736 |
| audio_scan | 14.805 |
| asr_timings | 11.066 |
| ast_timings | 24.424 |
| describe_scenes | 12.839 |
| summarize_scenes | 8.247 |
| synthesize_synopsis | 21.100 |
| make_embedding | 3.080 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.712 |
| branch_yolo_total | 9.661 |
| branch_audio_total | 50.304 |
