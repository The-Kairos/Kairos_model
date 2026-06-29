# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:14:12 UTC | WlJGA2-wZQ4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 208.599 | 0.785 | 104.228 | 10.962 | 14.360 | 13.889 | 3.829 |

## 2026-06-25 21:14:12 UTC | WlJGA2-wZQ4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WlJGA2-wZQ4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `208.599` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.785 |
| save_clips | - |
| sample_frames | 1.422 |
| caption_frames | 45.407 |
| sample_fps | 2.389 |
| detect_object_yolo | 9.905 |
| audio_scan | 14.015 |
| asr_timings | 57.562 |
| ast_timings | 32.642 |
| describe_scenes | 10.962 |
| summarize_scenes | 14.360 |
| synthesize_synopsis | 13.889 |
| make_embedding | 3.829 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.834 |
| branch_yolo_total | 12.300 |
| branch_audio_total | 104.228 |
