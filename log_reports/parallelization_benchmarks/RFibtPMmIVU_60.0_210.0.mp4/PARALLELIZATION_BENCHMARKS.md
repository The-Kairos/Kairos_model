# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 16:25:05 UTC | RFibtPMmIVU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 189.164 | 0.773 | 57.247 | 21.140 | 28.576 | 20.514 | 4.264 |

## 2026-06-25 16:25:05 UTC | RFibtPMmIVU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/RFibtPMmIVU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `189.164` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 1.099 |
| caption_frames | 42.746 |
| sample_fps | 2.259 |
| detect_object_yolo | 9.122 |
| audio_scan | 15.685 |
| asr_timings | 8.734 |
| ast_timings | 32.820 |
| describe_scenes | 21.140 |
| summarize_scenes | 28.576 |
| synthesize_synopsis | 20.514 |
| make_embedding | 4.264 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.851 |
| branch_yolo_total | 11.387 |
| branch_audio_total | 57.247 |
