# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:38:34 UTC | hfJvu-roZGQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.561 | 1.235 | 62.687 | 21.291 | 19.941 | 19.285 | 4.286 |

## 2026-06-26 06:38:34 UTC | hfJvu-roZGQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hfJvu-roZGQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.561` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.235 |
| save_clips | - |
| sample_frames | 0.943 |
| caption_frames | 46.719 |
| sample_fps | 0.983 |
| detect_object_yolo | 9.788 |
| audio_scan | 10.756 |
| asr_timings | 18.029 |
| ast_timings | 33.893 |
| describe_scenes | 21.291 |
| summarize_scenes | 19.941 |
| synthesize_synopsis | 19.285 |
| make_embedding | 4.286 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.667 |
| branch_yolo_total | 10.777 |
| branch_audio_total | 62.687 |
