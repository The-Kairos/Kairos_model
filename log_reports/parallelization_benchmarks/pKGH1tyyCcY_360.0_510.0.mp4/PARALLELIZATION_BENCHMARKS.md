# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:56:41 UTC | pKGH1tyyCcY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 217.231 | 0.780 | 110.643 | 13.465 | 18.029 | 7.625 | 4.104 |

## 2026-06-28 07:56:41 UTC | pKGH1tyyCcY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pKGH1tyyCcY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `217.231` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.207 |
| caption_frames | 47.884 |
| sample_fps | 2.336 |
| detect_object_yolo | 9.784 |
| audio_scan | 11.696 |
| asr_timings | 63.765 |
| ast_timings | 35.173 |
| describe_scenes | 13.465 |
| summarize_scenes | 18.029 |
| synthesize_synopsis | 7.625 |
| make_embedding | 4.104 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.097 |
| branch_yolo_total | 12.126 |
| branch_audio_total | 110.643 |
