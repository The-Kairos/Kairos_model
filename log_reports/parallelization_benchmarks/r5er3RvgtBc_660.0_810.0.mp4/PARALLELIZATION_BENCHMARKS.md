# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 09:23:23 UTC | r5er3RvgtBc_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 95.912 | 1.494 | 38.877 | 4.641 | 9.933 | 12.277 | 2.039 |

## 2026-06-28 09:23:23 UTC | r5er3RvgtBc_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/r5er3RvgtBc_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `95.912` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.494 |
| save_clips | - |
| sample_frames | 0.345 |
| caption_frames | 17.004 |
| sample_fps | 1.878 |
| detect_object_yolo | 6.049 |
| audio_scan | 15.975 |
| asr_timings | 10.257 |
| ast_timings | 12.637 |
| describe_scenes | 4.641 |
| summarize_scenes | 9.933 |
| synthesize_synopsis | 12.277 |
| make_embedding | 2.039 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 17.355 |
| branch_yolo_total | 7.932 |
| branch_audio_total | 38.877 |
