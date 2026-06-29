# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:46:33 UTC | MlXof8hF4ck_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 157.970 | 0.827 | 50.088 | 15.145 | 15.189 | 28.780 | 3.072 |

## 2026-06-25 09:46:33 UTC | MlXof8hF4ck_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MlXof8hF4ck_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `157.970` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.827 |
| save_clips | - |
| sample_frames | 0.743 |
| caption_frames | 32.912 |
| sample_fps | 2.077 |
| detect_object_yolo | 7.702 |
| audio_scan | 10.755 |
| asr_timings | 15.657 |
| ast_timings | 23.667 |
| describe_scenes | 15.145 |
| summarize_scenes | 15.189 |
| synthesize_synopsis | 28.780 |
| make_embedding | 3.072 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 33.661 |
| branch_yolo_total | 9.785 |
| branch_audio_total | 50.088 |
