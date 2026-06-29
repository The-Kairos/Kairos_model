# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:21:44 UTC | _Jq5O0D8q8M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.890 | 0.664 | 61.704 | 14.788 | 24.748 | 10.913 | 4.248 |

## 2026-06-25 23:21:44 UTC | _Jq5O0D8q8M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_Jq5O0D8q8M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.890` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.664 |
| save_clips | - |
| sample_frames | 1.425 |
| caption_frames | 46.941 |
| sample_fps | 2.272 |
| detect_object_yolo | 9.785 |
| audio_scan | 12.667 |
| asr_timings | 12.547 |
| ast_timings | 36.482 |
| describe_scenes | 14.788 |
| summarize_scenes | 24.748 |
| synthesize_synopsis | 10.913 |
| make_embedding | 4.248 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.371 |
| branch_yolo_total | 12.062 |
| branch_audio_total | 61.704 |
