# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:49:07 UTC | G-4tJ63X5vo_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.532 | 0.629 | 72.121 | 16.350 | 16.138 | 7.259 | 5.095 |

## 2026-06-25 00:49:07 UTC | G-4tJ63X5vo_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G-4tJ63X5vo_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.532` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.629 |
| save_clips | - |
| sample_frames | 1.242 |
| caption_frames | 55.821 |
| sample_fps | 2.247 |
| detect_object_yolo | 11.187 |
| audio_scan | 15.476 |
| asr_timings | 15.260 |
| ast_timings | 41.378 |
| describe_scenes | 16.350 |
| summarize_scenes | 16.138 |
| synthesize_synopsis | 7.259 |
| make_embedding | 5.095 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.069 |
| branch_yolo_total | 13.439 |
| branch_audio_total | 72.121 |
