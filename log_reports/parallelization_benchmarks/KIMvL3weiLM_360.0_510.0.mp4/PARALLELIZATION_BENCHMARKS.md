# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 06:12:32 UTC | KIMvL3weiLM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 183.652 | 0.667 | 59.420 | 13.170 | 26.465 | 12.052 | 4.561 |

## 2026-06-25 06:12:32 UTC | KIMvL3weiLM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/KIMvL3weiLM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `183.652` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.667 |
| save_clips | - |
| sample_frames | 1.505 |
| caption_frames | 51.655 |
| sample_fps | 2.321 |
| detect_object_yolo | 10.444 |
| audio_scan | 10.636 |
| asr_timings | 11.092 |
| ast_timings | 37.684 |
| describe_scenes | 13.170 |
| summarize_scenes | 26.465 |
| synthesize_synopsis | 12.052 |
| make_embedding | 4.561 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.166 |
| branch_yolo_total | 12.771 |
| branch_audio_total | 59.420 |
