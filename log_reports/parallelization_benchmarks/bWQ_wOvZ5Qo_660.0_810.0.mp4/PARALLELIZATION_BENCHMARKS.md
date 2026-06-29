# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:20:35 UTC | bWQ_wOvZ5Qo_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.983 | 0.835 | 60.723 | 13.479 | 13.891 | 8.764 | 4.156 |

## 2026-06-26 01:20:35 UTC | bWQ_wOvZ5Qo_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bWQ_wOvZ5Qo_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.983` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.835 |
| save_clips | - |
| sample_frames | 1.849 |
| caption_frames | 47.112 |
| sample_fps | 2.514 |
| detect_object_yolo | 9.232 |
| audio_scan | 14.991 |
| asr_timings | 11.049 |
| ast_timings | 34.676 |
| describe_scenes | 13.479 |
| summarize_scenes | 13.891 |
| synthesize_synopsis | 8.764 |
| make_embedding | 4.156 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 48.967 |
| branch_yolo_total | 11.751 |
| branch_audio_total | 60.723 |
