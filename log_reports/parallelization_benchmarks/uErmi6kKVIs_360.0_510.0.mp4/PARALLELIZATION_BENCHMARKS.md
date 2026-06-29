# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:44:58 UTC | uErmi6kKVIs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 102.989 | 0.762 | 53.958 | 4.912 | 4.443 | 10.035 | 2.061 |

## 2026-06-27 00:44:58 UTC | uErmi6kKVIs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uErmi6kKVIs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `102.989` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.762 |
| save_clips | - |
| sample_frames | 0.493 |
| caption_frames | 17.138 |
| sample_fps | 1.918 |
| detect_object_yolo | 5.870 |
| audio_scan | 13.814 |
| asr_timings | 26.915 |
| ast_timings | 13.220 |
| describe_scenes | 4.912 |
| summarize_scenes | 4.443 |
| synthesize_synopsis | 10.035 |
| make_embedding | 2.061 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.637 |
| branch_yolo_total | 7.795 |
| branch_audio_total | 53.958 |
