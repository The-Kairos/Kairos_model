# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 12:27:20 UTC | OkKEFnPQxPw_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 160.895 | 0.705 | 52.934 | 17.392 | 12.819 | 21.203 | 3.307 |

## 2026-06-25 12:27:20 UTC | OkKEFnPQxPw_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OkKEFnPQxPw_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `160.895` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.705 |
| save_clips | - |
| sample_frames | 1.370 |
| caption_frames | 38.390 |
| sample_fps | 2.218 |
| detect_object_yolo | 9.095 |
| audio_scan | 15.680 |
| asr_timings | 9.795 |
| ast_timings | 27.450 |
| describe_scenes | 17.392 |
| summarize_scenes | 12.819 |
| synthesize_synopsis | 21.203 |
| make_embedding | 3.307 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 39.767 |
| branch_yolo_total | 11.318 |
| branch_audio_total | 52.934 |
