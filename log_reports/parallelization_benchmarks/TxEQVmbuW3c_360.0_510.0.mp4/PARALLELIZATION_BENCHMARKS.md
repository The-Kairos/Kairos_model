# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:20:23 UTC | TxEQVmbuW3c_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.139 | 0.806 | 49.831 | 7.923 | 9.858 | 20.146 | 3.028 |

## 2026-06-25 18:20:23 UTC | TxEQVmbuW3c_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TxEQVmbuW3c_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.139` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.806 |
| save_clips | - |
| sample_frames | 0.627 |
| caption_frames | 30.496 |
| sample_fps | 2.059 |
| detect_object_yolo | 7.964 |
| audio_scan | 15.009 |
| asr_timings | 10.690 |
| ast_timings | 24.124 |
| describe_scenes | 7.923 |
| summarize_scenes | 9.858 |
| synthesize_synopsis | 20.146 |
| make_embedding | 3.028 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.129 |
| branch_yolo_total | 10.028 |
| branch_audio_total | 49.831 |
