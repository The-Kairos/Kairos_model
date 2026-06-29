# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 08:21:48 UTC | MH2nvDD_uo0_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.960 | 0.771 | 68.592 | 25.260 | 20.164 | 17.426 | 3.620 |

## 2026-06-25 08:21:48 UTC | MH2nvDD_uo0_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MH2nvDD_uo0_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.960` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.771 |
| save_clips | - |
| sample_frames | 1.133 |
| caption_frames | 42.089 |
| sample_fps | 2.238 |
| detect_object_yolo | 9.182 |
| audio_scan | 11.790 |
| asr_timings | 26.969 |
| ast_timings | 29.825 |
| describe_scenes | 25.260 |
| summarize_scenes | 20.164 |
| synthesize_synopsis | 17.426 |
| make_embedding | 3.620 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.228 |
| branch_yolo_total | 11.427 |
| branch_audio_total | 68.592 |
