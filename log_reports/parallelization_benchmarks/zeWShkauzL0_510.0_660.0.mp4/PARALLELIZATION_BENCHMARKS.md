# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 06:05:55 UTC | zeWShkauzL0_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.783 | 0.624 | 47.802 | 7.495 | 5.616 | 6.491 | 3.209 |

## 2026-06-27 06:05:55 UTC | zeWShkauzL0_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/zeWShkauzL0_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.783` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.624 |
| save_clips | - |
| sample_frames | 0.831 |
| caption_frames | 31.125 |
| sample_fps | 2.032 |
| detect_object_yolo | 8.167 |
| audio_scan | 12.866 |
| asr_timings | 10.332 |
| ast_timings | 24.595 |
| describe_scenes | 7.495 |
| summarize_scenes | 5.616 |
| synthesize_synopsis | 6.491 |
| make_embedding | 3.209 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 31.962 |
| branch_yolo_total | 10.204 |
| branch_audio_total | 47.802 |
