# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:26:34 UTC | di-kKBvebi8_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.704 | 0.784 | 44.447 | 10.676 | 5.929 | 10.811 | 2.802 |

## 2026-06-26 03:26:34 UTC | di-kKBvebi8_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/di-kKBvebi8_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.704` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.784 |
| save_clips | - |
| sample_frames | 0.835 |
| caption_frames | 28.291 |
| sample_fps | 2.079 |
| detect_object_yolo | 7.639 |
| audio_scan | 10.871 |
| asr_timings | 12.212 |
| ast_timings | 21.355 |
| describe_scenes | 10.676 |
| summarize_scenes | 5.929 |
| synthesize_synopsis | 10.811 |
| make_embedding | 2.802 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.131 |
| branch_yolo_total | 9.725 |
| branch_audio_total | 44.447 |
