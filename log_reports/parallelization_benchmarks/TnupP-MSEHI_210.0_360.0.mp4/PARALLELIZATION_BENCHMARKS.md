# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:00:14 UTC | TnupP-MSEHI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 172.453 | 0.778 | 60.039 | 13.930 | 20.465 | 12.217 | 4.292 |

## 2026-06-25 18:00:14 UTC | TnupP-MSEHI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TnupP-MSEHI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `172.453` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.778 |
| save_clips | - |
| sample_frames | 1.092 |
| caption_frames | 46.381 |
| sample_fps | 2.284 |
| detect_object_yolo | 9.569 |
| audio_scan | 12.837 |
| asr_timings | 11.685 |
| ast_timings | 35.509 |
| describe_scenes | 13.930 |
| summarize_scenes | 20.465 |
| synthesize_synopsis | 12.217 |
| make_embedding | 4.292 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.478 |
| branch_yolo_total | 11.859 |
| branch_audio_total | 60.039 |
