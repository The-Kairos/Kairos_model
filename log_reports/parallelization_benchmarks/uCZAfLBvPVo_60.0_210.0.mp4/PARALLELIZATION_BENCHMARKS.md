# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 00:40:13 UTC | uCZAfLBvPVo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 163.371 | 0.673 | 78.859 | 10.506 | 6.957 | 7.844 | 3.538 |

## 2026-06-27 00:40:13 UTC | uCZAfLBvPVo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uCZAfLBvPVo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `163.371` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.673 |
| save_clips | - |
| sample_frames | 1.330 |
| caption_frames | 40.676 |
| sample_fps | 2.211 |
| detect_object_yolo | 9.331 |
| audio_scan | 8.577 |
| asr_timings | 39.352 |
| ast_timings | 30.921 |
| describe_scenes | 10.506 |
| summarize_scenes | 6.957 |
| synthesize_synopsis | 7.844 |
| make_embedding | 3.538 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.012 |
| branch_yolo_total | 11.548 |
| branch_audio_total | 78.859 |
