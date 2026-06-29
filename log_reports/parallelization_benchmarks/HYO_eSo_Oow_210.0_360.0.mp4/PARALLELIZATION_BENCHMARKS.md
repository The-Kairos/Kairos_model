# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 03:38:47 UTC | HYO_eSo_Oow_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.508 | 0.872 | 56.149 | 13.967 | 21.939 | 9.871 | 3.926 |

## 2026-06-25 03:38:47 UTC | HYO_eSo_Oow_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/HYO_eSo_Oow_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.508` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.872 |
| save_clips | - |
| sample_frames | 1.357 |
| caption_frames | 45.212 |
| sample_fps | 2.413 |
| detect_object_yolo | 9.359 |
| audio_scan | 13.866 |
| asr_timings | 9.447 |
| ast_timings | 32.828 |
| describe_scenes | 13.967 |
| summarize_scenes | 21.939 |
| synthesize_synopsis | 9.871 |
| make_embedding | 3.926 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.576 |
| branch_yolo_total | 11.779 |
| branch_audio_total | 56.149 |
