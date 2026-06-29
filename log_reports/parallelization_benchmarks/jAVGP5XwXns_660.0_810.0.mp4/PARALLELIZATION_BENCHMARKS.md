# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 10:28:11 UTC | jAVGP5XwXns_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 133.609 | 1.286 | 43.270 | 11.598 | 6.926 | 36.723 | 2.292 |

## 2026-06-26 10:28:11 UTC | jAVGP5XwXns_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jAVGP5XwXns_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `133.609` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.286 |
| save_clips | - |
| sample_frames | 0.662 |
| caption_frames | 21.147 |
| sample_fps | 2.014 |
| detect_object_yolo | 6.275 |
| audio_scan | 16.076 |
| asr_timings | 11.277 |
| ast_timings | 15.908 |
| describe_scenes | 11.598 |
| summarize_scenes | 6.926 |
| synthesize_synopsis | 36.723 |
| make_embedding | 2.292 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.814 |
| branch_yolo_total | 8.294 |
| branch_audio_total | 43.270 |
