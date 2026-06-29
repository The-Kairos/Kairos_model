# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:18:08 UTC | TxEQVmbuW3c_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 111.945 | 0.806 | 39.773 | 8.177 | 13.028 | 14.791 | 2.250 |

## 2026-06-25 18:18:08 UTC | TxEQVmbuW3c_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/TxEQVmbuW3c_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `111.945` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.806 |
| save_clips | - |
| sample_frames | 0.490 |
| caption_frames | 22.363 |
| sample_fps | 1.968 |
| detect_object_yolo | 6.878 |
| audio_scan | 13.904 |
| asr_timings | 9.653 |
| ast_timings | 16.208 |
| describe_scenes | 8.177 |
| summarize_scenes | 13.028 |
| synthesize_synopsis | 14.791 |
| make_embedding | 2.250 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.858 |
| branch_yolo_total | 8.852 |
| branch_audio_total | 39.773 |
