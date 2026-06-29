# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 20:08:53 UTC | VtlI5SY9oQw_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 72.378 | 0.731 | 29.099 | 4.638 | 3.021 | 13.663 | 1.537 |

## 2026-06-25 20:08:53 UTC | VtlI5SY9oQw_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/VtlI5SY9oQw_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `72.378` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.731 |
| save_clips | - |
| sample_frames | 0.140 |
| caption_frames | 10.266 |
| sample_fps | 1.625 |
| detect_object_yolo | 6.288 |
| audio_scan | 11.724 |
| asr_timings | 10.186 |
| ast_timings | 7.180 |
| describe_scenes | 4.638 |
| summarize_scenes | 3.021 |
| synthesize_synopsis | 13.663 |
| make_embedding | 1.537 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.412 |
| branch_yolo_total | 7.918 |
| branch_audio_total | 29.099 |
