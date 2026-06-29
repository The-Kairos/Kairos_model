# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:20:36 UTC | kuM1lNdhYnI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.993 | 0.804 | 53.580 | 27.602 | 16.349 | 22.494 | 3.968 |

## 2026-06-26 14:20:36 UTC | kuM1lNdhYnI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kuM1lNdhYnI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.993` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.804 |
| save_clips | - |
| sample_frames | 1.143 |
| caption_frames | 43.717 |
| sample_fps | 2.349 |
| detect_object_yolo | 9.594 |
| audio_scan | 6.596 |
| asr_timings | 13.958 |
| ast_timings | 33.017 |
| describe_scenes | 27.602 |
| summarize_scenes | 16.349 |
| synthesize_synopsis | 22.494 |
| make_embedding | 3.968 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.866 |
| branch_yolo_total | 11.949 |
| branch_audio_total | 53.580 |
