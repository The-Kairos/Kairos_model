# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:19:27 UTC | RFibtPMmIVU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 148.566 | 0.767 | 46.178 | 11.945 | 17.233 | 23.996 | 3.090 |

## 2026-06-25 16:19:27 UTC | RFibtPMmIVU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/RFibtPMmIVU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `148.566` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.767 |
| save_clips | - |
| sample_frames | 1.000 |
| caption_frames | 32.651 |
| sample_fps | 2.173 |
| detect_object_yolo | 8.090 |
| audio_scan | 14.609 |
| asr_timings | 7.409 |
| ast_timings | 24.152 |
| describe_scenes | 11.945 |
| summarize_scenes | 17.233 |
| synthesize_synopsis | 23.996 |
| make_embedding | 3.090 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.657 |
| branch_yolo_total | 10.269 |
| branch_audio_total | 46.178 |
