# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 17:02:36 UTC | o0fdleH293A_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.960 | 0.784 | 47.640 | 8.579 | 7.552 | 10.367 | 2.858 |

## 2026-06-27 17:02:36 UTC | o0fdleH293A_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/o0fdleH293A_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.960` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.784 |
| save_clips | - |
| sample_frames | 0.726 |
| caption_frames | 27.667 |
| sample_fps | 2.061 |
| detect_object_yolo | 7.356 |
| audio_scan | 15.854 |
| asr_timings | 11.001 |
| ast_timings | 20.777 |
| describe_scenes | 8.579 |
| summarize_scenes | 7.552 |
| synthesize_synopsis | 10.367 |
| make_embedding | 2.858 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 28.399 |
| branch_yolo_total | 9.422 |
| branch_audio_total | 47.640 |
