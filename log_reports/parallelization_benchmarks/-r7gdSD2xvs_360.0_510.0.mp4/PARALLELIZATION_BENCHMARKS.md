# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 12:58:05 UTC | -r7gdSD2xvs_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 94.208 | 0.657 | 40.014 | 8.062 | 4.649 | 7.237 | 2.295 |

## 2026-06-27 12:58:05 UTC | -r7gdSD2xvs_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-r7gdSD2xvs_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `94.208` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.657 |
| save_clips | - |
| sample_frames | 0.509 |
| caption_frames | 20.780 |
| sample_fps | 1.806 |
| detect_object_yolo | 6.789 |
| audio_scan | 14.962 |
| asr_timings | 9.526 |
| ast_timings | 15.517 |
| describe_scenes | 8.062 |
| summarize_scenes | 4.649 |
| synthesize_synopsis | 7.237 |
| make_embedding | 2.295 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.295 |
| branch_yolo_total | 8.601 |
| branch_audio_total | 40.014 |
