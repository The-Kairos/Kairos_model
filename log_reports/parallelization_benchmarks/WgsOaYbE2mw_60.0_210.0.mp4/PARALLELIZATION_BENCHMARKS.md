# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 21:01:11 UTC | WgsOaYbE2mw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.889 | 0.797 | 57.046 | 17.981 | 18.642 | 12.230 | 4.223 |

## 2026-06-25 21:01:11 UTC | WgsOaYbE2mw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/WgsOaYbE2mw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.889` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.797 |
| save_clips | - |
| sample_frames | 1.441 |
| caption_frames | 48.095 |
| sample_fps | 2.449 |
| detect_object_yolo | 9.582 |
| audio_scan | 11.793 |
| asr_timings | 8.743 |
| ast_timings | 36.502 |
| describe_scenes | 17.981 |
| summarize_scenes | 18.642 |
| synthesize_synopsis | 12.230 |
| make_embedding | 4.223 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.542 |
| branch_yolo_total | 12.037 |
| branch_audio_total | 57.046 |
