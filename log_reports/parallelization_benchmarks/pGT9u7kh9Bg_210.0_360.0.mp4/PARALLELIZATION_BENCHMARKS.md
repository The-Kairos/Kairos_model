# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:46:30 UTC | pGT9u7kh9Bg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 199.729 | 0.688 | 73.106 | 17.289 | 12.975 | 8.288 | 6.040 |

## 2026-06-28 07:46:30 UTC | pGT9u7kh9Bg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pGT9u7kh9Bg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.729` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.688 |
| save_clips | - |
| sample_frames | 1.545 |
| caption_frames | 64.218 |
| sample_fps | 2.404 |
| detect_object_yolo | 11.774 |
| audio_scan | 14.832 |
| asr_timings | 8.633 |
| ast_timings | 49.632 |
| describe_scenes | 17.289 |
| summarize_scenes | 12.975 |
| synthesize_synopsis | 8.288 |
| make_embedding | 6.040 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 65.769 |
| branch_yolo_total | 14.184 |
| branch_audio_total | 73.106 |
