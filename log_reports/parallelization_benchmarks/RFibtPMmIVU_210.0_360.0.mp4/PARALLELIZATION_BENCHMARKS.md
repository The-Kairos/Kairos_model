# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:16:58 UTC | RFibtPMmIVU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.560 | 0.802 | 53.579 | 11.945 | 8.910 | 20.397 | 3.683 |

## 2026-06-25 16:16:58 UTC | RFibtPMmIVU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/RFibtPMmIVU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.560` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.802 |
| save_clips | - |
| sample_frames | 1.288 |
| caption_frames | 40.412 |
| sample_fps | 2.308 |
| detect_object_yolo | 8.761 |
| audio_scan | 15.800 |
| asr_timings | 7.598 |
| ast_timings | 30.172 |
| describe_scenes | 11.945 |
| summarize_scenes | 8.910 |
| synthesize_synopsis | 20.397 |
| make_embedding | 3.683 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.706 |
| branch_yolo_total | 11.075 |
| branch_audio_total | 53.579 |
