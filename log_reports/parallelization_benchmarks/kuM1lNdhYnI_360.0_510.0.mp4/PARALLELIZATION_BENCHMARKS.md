# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:23:38 UTC | kuM1lNdhYnI_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 180.258 | 0.816 | 50.926 | 23.980 | 26.207 | 19.165 | 3.579 |

## 2026-06-26 14:23:38 UTC | kuM1lNdhYnI_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kuM1lNdhYnI_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `180.258` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.816 |
| save_clips | - |
| sample_frames | 1.051 |
| caption_frames | 41.378 |
| sample_fps | 2.328 |
| detect_object_yolo | 9.332 |
| audio_scan | 8.780 |
| asr_timings | 12.468 |
| ast_timings | 29.670 |
| describe_scenes | 23.980 |
| summarize_scenes | 26.207 |
| synthesize_synopsis | 19.165 |
| make_embedding | 3.579 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.436 |
| branch_yolo_total | 11.666 |
| branch_audio_total | 50.926 |
