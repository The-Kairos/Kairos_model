# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 22:59:54 UTC | DV7_zeiQhdU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.096 | 0.788 | 75.268 | 19.631 | 10.726 | 15.288 | 6.064 |

## 2026-06-24 22:59:54 UTC | DV7_zeiQhdU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/DV7_zeiQhdU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.096` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 1.600 |
| caption_frames | 69.395 |
| sample_fps | 2.598 |
| detect_object_yolo | 12.321 |
| audio_scan | 15.970 |
| asr_timings | 9.994 |
| ast_timings | 49.295 |
| describe_scenes | 19.631 |
| summarize_scenes | 10.726 |
| synthesize_synopsis | 15.288 |
| make_embedding | 6.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 71.002 |
| branch_yolo_total | 14.925 |
| branch_audio_total | 75.268 |
