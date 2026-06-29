# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:59:08 UTC | UO0kY5njBvo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 191.502 | 0.812 | 59.693 | 27.840 | 12.016 | 21.063 | 4.533 |

## 2026-06-25 18:59:08 UTC | UO0kY5njBvo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UO0kY5njBvo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `191.502` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.812 |
| save_clips | - |
| sample_frames | 1.590 |
| caption_frames | 49.955 |
| sample_fps | 2.491 |
| detect_object_yolo | 10.084 |
| audio_scan | 12.769 |
| asr_timings | 9.103 |
| ast_timings | 37.813 |
| describe_scenes | 27.840 |
| summarize_scenes | 12.016 |
| synthesize_synopsis | 21.063 |
| make_embedding | 4.533 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.551 |
| branch_yolo_total | 12.581 |
| branch_audio_total | 59.693 |
