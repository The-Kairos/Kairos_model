# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 05:38:46 UTC | yvx5SmeFdjM_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 158.329 | 0.760 | 65.889 | 9.690 | 9.950 | 9.925 | 3.825 |

## 2026-06-27 05:38:46 UTC | yvx5SmeFdjM_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/yvx5SmeFdjM_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `158.329` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.760 |
| save_clips | - |
| sample_frames | 1.360 |
| caption_frames | 43.778 |
| sample_fps | 2.356 |
| detect_object_yolo | 9.408 |
| audio_scan | 12.854 |
| asr_timings | 20.558 |
| ast_timings | 32.468 |
| describe_scenes | 9.690 |
| summarize_scenes | 9.950 |
| synthesize_synopsis | 9.925 |
| make_embedding | 3.825 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 45.144 |
| branch_yolo_total | 11.770 |
| branch_audio_total | 65.889 |
