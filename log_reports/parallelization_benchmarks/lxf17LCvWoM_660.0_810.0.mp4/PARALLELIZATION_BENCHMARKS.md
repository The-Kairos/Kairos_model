# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:32:01 UTC | lxf17LCvWoM_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 218.914 | 0.794 | 67.017 | 27.163 | 33.011 | 15.633 | 5.045 |

## 2026-06-26 17:32:01 UTC | lxf17LCvWoM_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lxf17LCvWoM_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `218.914` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.794 |
| save_clips | - |
| sample_frames | 1.330 |
| caption_frames | 54.064 |
| sample_fps | 2.412 |
| detect_object_yolo | 11.016 |
| audio_scan | 13.970 |
| asr_timings | 11.408 |
| ast_timings | 41.630 |
| describe_scenes | 27.163 |
| summarize_scenes | 33.011 |
| synthesize_synopsis | 15.633 |
| make_embedding | 5.045 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.400 |
| branch_yolo_total | 13.434 |
| branch_audio_total | 67.017 |
