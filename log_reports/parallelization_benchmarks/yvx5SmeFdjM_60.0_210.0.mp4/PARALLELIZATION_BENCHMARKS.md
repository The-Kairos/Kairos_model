# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:42:04 UTC | yvx5SmeFdjM_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 196.953 | 0.760 | 93.176 | 12.654 | 13.609 | 6.579 | 4.738 |

## 2026-06-27 05:42:04 UTC | yvx5SmeFdjM_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yvx5SmeFdjM_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `196.953` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 1.303 |
| caption_frames | 50.302 |
| sample_fps | 2.389 |
| detect_object_yolo | 10.044 |
| audio_scan | 15.996 |
| asr_timings | 38.789 |
| ast_timings | 38.382 |
| describe_scenes | 12.654 |
| summarize_scenes | 13.609 |
| synthesize_synopsis | 6.579 |
| make_embedding | 4.738 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.611 |
| branch_yolo_total | 12.438 |
| branch_audio_total | 93.176 |
