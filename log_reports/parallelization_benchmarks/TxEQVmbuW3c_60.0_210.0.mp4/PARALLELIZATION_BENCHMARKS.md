# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:24:24 UTC | TxEQVmbuW3c_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 72.846 | 0.788 | 29.347 | 4.474 | 5.970 | 14.328 | 1.328 |

## 2026-06-25 18:24:24 UTC | TxEQVmbuW3c_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TxEQVmbuW3c_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `72.846` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 0.085 |
| caption_frames | 7.976 |
| sample_fps | 1.750 |
| detect_object_yolo | 5.415 |
| audio_scan | 13.928 |
| asr_timings | 10.940 |
| ast_timings | 4.470 |
| describe_scenes | 4.474 |
| summarize_scenes | 5.970 |
| synthesize_synopsis | 14.328 |
| make_embedding | 1.328 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 8.067 |
| branch_yolo_total | 7.171 |
| branch_audio_total | 29.347 |
