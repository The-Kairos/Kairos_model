# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:46:45 UTC | 6ieHWdhGczs_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 213.973 | 0.719 | 101.434 | 17.731 | 13.716 | 19.083 | 3.932 |

## 2026-06-24 12:46:45 UTC | 6ieHWdhGczs_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6ieHWdhGczs_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `213.973` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.719 |
| save_clips | - |
| sample_frames | 1.649 |
| caption_frames | 42.488 |
| sample_fps | 2.393 |
| detect_object_yolo | 9.434 |
| audio_scan | 11.642 |
| asr_timings | 57.938 |
| ast_timings | 31.845 |
| describe_scenes | 17.731 |
| summarize_scenes | 13.716 |
| synthesize_synopsis | 19.083 |
| make_embedding | 3.932 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.143 |
| branch_yolo_total | 11.832 |
| branch_audio_total | 101.434 |
