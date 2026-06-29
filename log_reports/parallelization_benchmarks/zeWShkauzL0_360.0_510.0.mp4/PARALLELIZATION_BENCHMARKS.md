# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:03:59 UTC | zeWShkauzL0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 184.499 | 0.670 | 67.206 | 17.362 | 10.998 | 7.992 | 5.378 |

## 2026-06-27 06:03:59 UTC | zeWShkauzL0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeWShkauzL0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `184.499` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.670 |
| save_clips | - |
| sample_frames | 1.521 |
| caption_frames | 58.281 |
| sample_fps | 2.420 |
| detect_object_yolo | 11.238 |
| audio_scan | 13.906 |
| asr_timings | 8.425 |
| ast_timings | 44.866 |
| describe_scenes | 17.362 |
| summarize_scenes | 10.998 |
| synthesize_synopsis | 7.992 |
| make_embedding | 5.378 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.808 |
| branch_yolo_total | 13.664 |
| branch_audio_total | 67.206 |
