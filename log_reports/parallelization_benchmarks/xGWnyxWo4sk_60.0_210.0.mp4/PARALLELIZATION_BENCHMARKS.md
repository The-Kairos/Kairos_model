# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:46:45 UTC | xGWnyxWo4sk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 135.726 | 0.800 | 52.422 | 11.105 | 9.955 | 7.695 | 3.398 |

## 2026-06-27 03:46:45 UTC | xGWnyxWo4sk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xGWnyxWo4sk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `135.726` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.800 |
| save_clips | - |
| sample_frames | 1.166 |
| caption_frames | 36.828 |
| sample_fps | 2.277 |
| detect_object_yolo | 8.639 |
| audio_scan | 16.279 |
| asr_timings | 8.636 |
| ast_timings | 27.498 |
| describe_scenes | 11.105 |
| summarize_scenes | 9.955 |
| synthesize_synopsis | 7.695 |
| make_embedding | 3.398 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 38.000 |
| branch_yolo_total | 10.922 |
| branch_audio_total | 52.422 |
