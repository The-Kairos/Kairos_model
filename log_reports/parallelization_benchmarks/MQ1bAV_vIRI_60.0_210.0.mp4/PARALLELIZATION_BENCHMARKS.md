# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:36:09 UTC | MQ1bAV_vIRI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 153.449 | 0.787 | 45.440 | 17.443 | 17.899 | 29.308 | 2.829 |

## 2026-06-25 09:36:09 UTC | MQ1bAV_vIRI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MQ1bAV_vIRI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `153.449` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 0.630 |
| caption_frames | 28.475 |
| sample_fps | 2.048 |
| detect_object_yolo | 7.167 |
| audio_scan | 16.101 |
| asr_timings | 7.715 |
| ast_timings | 21.615 |
| describe_scenes | 17.443 |
| summarize_scenes | 17.899 |
| synthesize_synopsis | 29.308 |
| make_embedding | 2.829 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.112 |
| branch_yolo_total | 9.221 |
| branch_audio_total | 45.440 |
