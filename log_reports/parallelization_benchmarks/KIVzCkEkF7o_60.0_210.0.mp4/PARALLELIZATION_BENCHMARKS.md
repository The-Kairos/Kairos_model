# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:25:26 UTC | KIVzCkEkF7o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.897 | 0.811 | 58.282 | 18.606 | 19.164 | 20.560 | 4.182 |

## 2026-06-25 06:25:26 UTC | KIVzCkEkF7o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KIVzCkEkF7o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.897` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.811 |
| save_clips | - |
| sample_frames | 1.123 |
| caption_frames | 48.818 |
| sample_fps | 2.294 |
| detect_object_yolo | 9.660 |
| audio_scan | 14.830 |
| asr_timings | 7.876 |
| ast_timings | 35.568 |
| describe_scenes | 18.606 |
| summarize_scenes | 19.164 |
| synthesize_synopsis | 20.560 |
| make_embedding | 4.182 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.947 |
| branch_yolo_total | 11.960 |
| branch_audio_total | 58.282 |
