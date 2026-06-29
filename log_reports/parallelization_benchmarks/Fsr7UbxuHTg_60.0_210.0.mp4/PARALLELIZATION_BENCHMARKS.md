# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:45:56 UTC | Fsr7UbxuHTg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.712 | 0.808 | 56.213 | 11.394 | 13.600 | 10.836 | 3.326 |

## 2026-06-25 00:45:56 UTC | Fsr7UbxuHTg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Fsr7UbxuHTg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.712` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.808 |
| save_clips | - |
| sample_frames | 0.983 |
| caption_frames | 36.343 |
| sample_fps | 2.219 |
| detect_object_yolo | 8.560 |
| audio_scan | 16.121 |
| asr_timings | 12.731 |
| ast_timings | 27.352 |
| describe_scenes | 11.394 |
| summarize_scenes | 13.600 |
| synthesize_synopsis | 10.836 |
| make_embedding | 3.326 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.332 |
| branch_yolo_total | 10.785 |
| branch_audio_total | 56.213 |
