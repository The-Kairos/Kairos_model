# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 09:02:58 UTC | qpb2Z8lpPhA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.471 | 0.613 | 58.190 | 11.703 | 8.141 | 5.899 | 3.839 |

## 2026-06-28 09:02:58 UTC | qpb2Z8lpPhA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/qpb2Z8lpPhA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.471` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.613 |
| save_clips | - |
| sample_frames | 1.372 |
| caption_frames | 41.744 |
| sample_fps | 2.214 |
| detect_object_yolo | 9.373 |
| audio_scan | 13.867 |
| asr_timings | 11.394 |
| ast_timings | 32.922 |
| describe_scenes | 11.703 |
| summarize_scenes | 8.141 |
| synthesize_synopsis | 5.899 |
| make_embedding | 3.839 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 43.121 |
| branch_yolo_total | 11.594 |
| branch_audio_total | 58.190 |
