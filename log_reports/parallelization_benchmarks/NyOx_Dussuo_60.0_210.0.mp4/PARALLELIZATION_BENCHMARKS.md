# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:17:54 UTC | NyOx_Dussuo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.417 | 0.779 | 46.997 | 11.540 | 9.486 | 20.115 | 2.772 |

## 2026-06-25 11:17:54 UTC | NyOx_Dussuo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/NyOx_Dussuo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.417` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 0.847 |
| caption_frames | 30.717 |
| sample_fps | 2.088 |
| detect_object_yolo | 7.673 |
| audio_scan | 15.078 |
| asr_timings | 10.394 |
| ast_timings | 21.516 |
| describe_scenes | 11.540 |
| summarize_scenes | 9.486 |
| synthesize_synopsis | 20.115 |
| make_embedding | 2.772 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.570 |
| branch_yolo_total | 9.767 |
| branch_audio_total | 46.997 |
