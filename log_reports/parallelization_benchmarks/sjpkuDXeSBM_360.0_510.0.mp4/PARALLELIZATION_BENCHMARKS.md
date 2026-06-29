# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 20:09:50 UTC | sjpkuDXeSBM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 182.883 | 0.812 | 62.502 | 18.593 | 9.755 | 21.139 | 4.407 |

## 2026-06-26 20:09:50 UTC | sjpkuDXeSBM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/sjpkuDXeSBM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `182.883` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.539 |
| caption_frames | 50.102 |
| sample_fps | 2.430 |
| detect_object_yolo | 10.167 |
| audio_scan | 12.937 |
| asr_timings | 11.208 |
| ast_timings | 38.348 |
| describe_scenes | 18.593 |
| summarize_scenes | 9.755 |
| synthesize_synopsis | 21.139 |
| make_embedding | 4.407 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.646 |
| branch_yolo_total | 12.603 |
| branch_audio_total | 62.502 |
