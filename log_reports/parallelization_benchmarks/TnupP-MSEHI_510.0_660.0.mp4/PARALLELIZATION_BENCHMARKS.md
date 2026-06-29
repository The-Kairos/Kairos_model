# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:06:10 UTC | TnupP-MSEHI_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.157 | 0.792 | 60.992 | 16.616 | 13.332 | 11.632 | 4.171 |

## 2026-06-25 18:06:10 UTC | TnupP-MSEHI_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TnupP-MSEHI_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.157` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.176 |
| caption_frames | 47.787 |
| sample_fps | 2.344 |
| detect_object_yolo | 9.911 |
| audio_scan | 13.753 |
| asr_timings | 11.181 |
| ast_timings | 36.049 |
| describe_scenes | 16.616 |
| summarize_scenes | 13.332 |
| synthesize_synopsis | 11.632 |
| make_embedding | 4.171 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.970 |
| branch_yolo_total | 12.261 |
| branch_audio_total | 60.992 |
