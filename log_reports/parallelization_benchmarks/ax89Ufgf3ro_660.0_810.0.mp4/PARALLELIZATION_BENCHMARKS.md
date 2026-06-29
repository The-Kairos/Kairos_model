# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 00:37:05 UTC | ax89Ufgf3ro_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 162.247 | 0.647 | 61.025 | 14.818 | 9.410 | 11.253 | 4.152 |

## 2026-06-26 00:37:05 UTC | ax89Ufgf3ro_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ax89Ufgf3ro_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `162.247` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.647 |
| save_clips | - |
| sample_frames | 1.435 |
| caption_frames | 46.028 |
| sample_fps | 2.296 |
| detect_object_yolo | 9.750 |
| audio_scan | 16.142 |
| asr_timings | 9.706 |
| ast_timings | 35.169 |
| describe_scenes | 14.818 |
| summarize_scenes | 9.410 |
| synthesize_synopsis | 11.253 |
| make_embedding | 4.152 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.469 |
| branch_yolo_total | 12.052 |
| branch_audio_total | 61.025 |
