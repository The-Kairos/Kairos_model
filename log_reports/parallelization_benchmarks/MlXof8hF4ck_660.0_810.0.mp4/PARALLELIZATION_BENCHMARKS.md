# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:58:47 UTC | MlXof8hF4ck_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 141.046 | 0.788 | 41.732 | 9.414 | 26.160 | 23.783 | 2.556 |

## 2026-06-25 09:58:47 UTC | MlXof8hF4ck_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MlXof8hF4ck_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `141.046` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 0.485 |
| caption_frames | 25.858 |
| sample_fps | 2.010 |
| detect_object_yolo | 6.833 |
| audio_scan | 14.113 |
| asr_timings | 9.996 |
| ast_timings | 17.613 |
| describe_scenes | 9.414 |
| summarize_scenes | 26.160 |
| synthesize_synopsis | 23.783 |
| make_embedding | 2.556 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.349 |
| branch_yolo_total | 8.848 |
| branch_audio_total | 41.732 |
