# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 14:18:19 UTC | 7YEQ0iDR4sw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 120.815 | 0.737 | 39.519 | 10.126 | 13.288 | 22.796 | 2.395 |

## 2026-06-24 14:18:19 UTC | 7YEQ0iDR4sw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/7YEQ0iDR4sw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `120.815` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.737 |
| save_clips | - |
| sample_frames | 0.635 |
| caption_frames | 21.206 |
| sample_fps | 1.864 |
| detect_object_yolo | 6.851 |
| audio_scan | 13.889 |
| asr_timings | 9.904 |
| ast_timings | 15.717 |
| describe_scenes | 10.126 |
| summarize_scenes | 13.288 |
| synthesize_synopsis | 22.796 |
| make_embedding | 2.395 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 21.846 |
| branch_yolo_total | 8.721 |
| branch_audio_total | 39.519 |
