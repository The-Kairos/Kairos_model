# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 14:26:41 UTC | kuM1lNdhYnI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.525 | 0.808 | 53.570 | 23.498 | 20.350 | 23.111 | 3.863 |

## 2026-06-26 14:26:41 UTC | kuM1lNdhYnI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kuM1lNdhYnI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.525` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.808 |
| save_clips | - |
| sample_frames | 1.078 |
| caption_frames | 42.994 |
| sample_fps | 2.342 |
| detect_object_yolo | 9.472 |
| audio_scan | 6.547 |
| asr_timings | 13.745 |
| ast_timings | 33.270 |
| describe_scenes | 23.498 |
| summarize_scenes | 20.350 |
| synthesize_synopsis | 23.111 |
| make_embedding | 3.863 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.078 |
| branch_yolo_total | 11.819 |
| branch_audio_total | 53.570 |
