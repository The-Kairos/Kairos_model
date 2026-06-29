# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 02:21:26 UTC | vbqZqioIK1c_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 134.880 | 0.630 | 49.613 | 8.239 | 11.140 | 14.433 | 3.307 |

## 2026-06-27 02:21:26 UTC | vbqZqioIK1c_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/vbqZqioIK1c_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `134.880` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.630 |
| save_clips | - |
| sample_frames | 0.764 |
| caption_frames | 34.875 |
| sample_fps | 1.984 |
| detect_object_yolo | 8.501 |
| audio_scan | 10.810 |
| asr_timings | 12.626 |
| ast_timings | 26.168 |
| describe_scenes | 8.239 |
| summarize_scenes | 11.140 |
| synthesize_synopsis | 14.433 |
| make_embedding | 3.307 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.645 |
| branch_yolo_total | 10.490 |
| branch_audio_total | 49.613 |
