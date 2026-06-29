# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:08:28 UTC | TnupP-MSEHI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.497 | 0.768 | 51.556 | 10.776 | 8.512 | 13.540 | 3.297 |

## 2026-06-25 18:08:28 UTC | TnupP-MSEHI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TnupP-MSEHI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.497` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.768 |
| save_clips | - |
| sample_frames | 0.885 |
| caption_frames | 35.111 |
| sample_fps | 2.167 |
| detect_object_yolo | 8.487 |
| audio_scan | 14.928 |
| asr_timings | 9.517 |
| ast_timings | 27.103 |
| describe_scenes | 10.776 |
| summarize_scenes | 8.512 |
| synthesize_synopsis | 13.540 |
| make_embedding | 3.297 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.001 |
| branch_yolo_total | 10.660 |
| branch_audio_total | 51.556 |
