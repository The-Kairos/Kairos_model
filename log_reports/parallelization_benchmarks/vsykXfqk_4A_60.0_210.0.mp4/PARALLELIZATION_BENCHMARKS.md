# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:44:05 UTC | vsykXfqk_4A_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.113 | 0.790 | 66.606 | 16.120 | 16.237 | 8.502 | 5.899 |

## 2026-06-27 02:44:05 UTC | vsykXfqk_4A_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vsykXfqk_4A_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.113` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.790 |
| save_clips | - |
| sample_frames | 1.467 |
| caption_frames | 57.834 |
| sample_fps | 2.468 |
| detect_object_yolo | 10.784 |
| audio_scan | 10.819 |
| asr_timings | 11.455 |
| ast_timings | 44.324 |
| describe_scenes | 16.120 |
| summarize_scenes | 16.237 |
| synthesize_synopsis | 8.502 |
| make_embedding | 5.899 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.307 |
| branch_yolo_total | 13.257 |
| branch_audio_total | 66.606 |
