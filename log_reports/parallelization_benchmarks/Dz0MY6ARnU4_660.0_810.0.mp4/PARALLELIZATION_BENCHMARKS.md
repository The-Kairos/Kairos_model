# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 23:24:54 UTC | Dz0MY6ARnU4_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 123.833 | 0.678 | 84.771 | 2.865 | 2.576 | 7.296 | 1.800 |

## 2026-06-24 23:24:54 UTC | Dz0MY6ARnU4_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Dz0MY6ARnU4_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `123.833` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.678 |
| save_clips | - |
| sample_frames | 0.331 |
| caption_frames | 13.918 |
| sample_fps | 1.719 |
| detect_object_yolo | 6.459 |
| audio_scan | 14.944 |
| asr_timings | 59.453 |
| ast_timings | 10.364 |
| describe_scenes | 2.865 |
| summarize_scenes | 2.576 |
| synthesize_synopsis | 7.296 |
| make_embedding | 1.800 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 14.255 |
| branch_yolo_total | 8.183 |
| branch_audio_total | 84.771 |
