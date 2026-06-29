# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 20:39:20 UTC | CYGNn8t90Wk_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 207.067 | 0.635 | 74.004 | 16.188 | 20.628 | 13.064 | 5.728 |

## 2026-06-24 20:39:20 UTC | CYGNn8t90Wk_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/CYGNn8t90Wk_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `207.067` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.635 |
| save_clips | - |
| sample_frames | 1.441 |
| caption_frames | 60.364 |
| sample_fps | 2.354 |
| detect_object_yolo | 11.259 |
| audio_scan | 9.652 |
| asr_timings | 17.706 |
| ast_timings | 46.637 |
| describe_scenes | 16.188 |
| summarize_scenes | 20.628 |
| synthesize_synopsis | 13.064 |
| make_embedding | 5.728 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.812 |
| branch_yolo_total | 13.618 |
| branch_audio_total | 74.004 |
