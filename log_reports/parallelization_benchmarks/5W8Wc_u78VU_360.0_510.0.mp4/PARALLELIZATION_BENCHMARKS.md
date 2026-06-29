# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:07:33 UTC | 5W8Wc_u78VU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.178 | 0.690 | 69.233 | 23.325 | 20.900 | 13.475 | 6.300 |

## 2026-06-24 12:07:33 UTC | 5W8Wc_u78VU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5W8Wc_u78VU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.178` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.690 |
| save_clips | - |
| sample_frames | 1.508 |
| caption_frames | 63.944 |
| sample_fps | 2.450 |
| detect_object_yolo | 11.951 |
| audio_scan | 11.675 |
| asr_timings | 8.310 |
| ast_timings | 49.240 |
| describe_scenes | 23.325 |
| summarize_scenes | 20.900 |
| synthesize_synopsis | 13.475 |
| make_embedding | 6.300 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.458 |
| branch_yolo_total | 14.407 |
| branch_audio_total | 69.233 |
