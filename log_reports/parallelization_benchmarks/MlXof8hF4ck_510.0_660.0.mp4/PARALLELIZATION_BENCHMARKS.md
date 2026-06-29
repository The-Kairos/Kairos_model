# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:53:12 UTC | MlXof8hF4ck_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 178.107 | 0.792 | 62.918 | 19.382 | 13.586 | 14.375 | 4.104 |

## 2026-06-25 09:53:12 UTC | MlXof8hF4ck_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MlXof8hF4ck_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `178.107` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.792 |
| save_clips | - |
| sample_frames | 1.035 |
| caption_frames | 48.261 |
| sample_fps | 2.272 |
| detect_object_yolo | 9.924 |
| audio_scan | 15.096 |
| asr_timings | 13.115 |
| ast_timings | 34.699 |
| describe_scenes | 19.382 |
| summarize_scenes | 13.586 |
| synthesize_synopsis | 14.375 |
| make_embedding | 4.104 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.302 |
| branch_yolo_total | 12.201 |
| branch_audio_total | 62.918 |
