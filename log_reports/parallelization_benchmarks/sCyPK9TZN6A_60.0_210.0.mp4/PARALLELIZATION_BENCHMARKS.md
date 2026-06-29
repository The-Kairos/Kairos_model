# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 19:42:10 UTC | sCyPK9TZN6A_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 103.326 | 0.791 | 35.409 | 8.234 | 6.068 | 17.895 | 2.314 |

## 2026-06-26 19:42:10 UTC | sCyPK9TZN6A_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sCyPK9TZN6A_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `103.326` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 0.560 |
| caption_frames | 21.359 |
| sample_fps | 1.990 |
| detect_object_yolo | 7.235 |
| audio_scan | 8.587 |
| asr_timings | 11.204 |
| ast_timings | 15.609 |
| describe_scenes | 8.234 |
| summarize_scenes | 6.068 |
| synthesize_synopsis | 17.895 |
| make_embedding | 2.314 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.925 |
| branch_yolo_total | 9.230 |
| branch_audio_total | 35.409 |
