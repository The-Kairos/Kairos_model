# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 06:40:44 UTC | hfJvu-roZGQ_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 129.202 | 1.034 | 45.813 | 17.865 | 11.957 | 18.289 | 2.335 |

## 2026-06-26 06:40:44 UTC | hfJvu-roZGQ_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hfJvu-roZGQ_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `129.202` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.034 |
| save_clips | - |
| sample_frames | 0.428 |
| caption_frames | 23.209 |
| sample_fps | 0.647 |
| detect_object_yolo | 6.236 |
| audio_scan | 13.444 |
| asr_timings | 16.949 |
| ast_timings | 15.411 |
| describe_scenes | 17.865 |
| summarize_scenes | 11.957 |
| synthesize_synopsis | 18.289 |
| make_embedding | 2.335 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 23.643 |
| branch_yolo_total | 6.889 |
| branch_audio_total | 45.813 |
