# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 22:15:26 UTC | Z3F41bP1Lcs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 124.430 | 0.663 | 41.203 | 10.725 | 16.091 | 10.524 | 2.806 |

## 2026-06-25 22:15:26 UTC | Z3F41bP1Lcs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Z3F41bP1Lcs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `124.430` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.663 |
| save_clips | - |
| sample_frames | 0.841 |
| caption_frames | 29.945 |
| sample_fps | 2.021 |
| detect_object_yolo | 8.198 |
| audio_scan | 6.471 |
| asr_timings | 12.904 |
| ast_timings | 21.820 |
| describe_scenes | 10.725 |
| summarize_scenes | 16.091 |
| synthesize_synopsis | 10.524 |
| make_embedding | 2.806 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.792 |
| branch_yolo_total | 10.225 |
| branch_audio_total | 41.203 |
