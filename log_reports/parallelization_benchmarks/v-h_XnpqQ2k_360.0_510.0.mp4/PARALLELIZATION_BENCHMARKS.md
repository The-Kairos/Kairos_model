# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:52:38 UTC | v-h_XnpqQ2k_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 110.284 | 0.550 | 39.911 | 7.769 | 6.965 | 12.921 | 2.735 |

## 2026-06-27 01:52:38 UTC | v-h_XnpqQ2k_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/v-h_XnpqQ2k_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `110.284` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.550 |
| save_clips | - |
| sample_frames | 0.629 |
| caption_frames | 28.871 |
| sample_fps | 1.689 |
| detect_object_yolo | 6.836 |
| audio_scan | 9.037 |
| asr_timings | 8.418 |
| ast_timings | 22.448 |
| describe_scenes | 7.769 |
| summarize_scenes | 6.965 |
| synthesize_synopsis | 12.921 |
| make_embedding | 2.735 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.505 |
| branch_yolo_total | 8.530 |
| branch_audio_total | 39.911 |
