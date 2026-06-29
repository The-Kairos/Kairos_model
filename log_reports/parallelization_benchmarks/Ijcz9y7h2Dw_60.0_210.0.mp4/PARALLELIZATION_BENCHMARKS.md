# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:56:42 UTC | Ijcz9y7h2Dw_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 82.615 | 0.694 | 26.143 | 4.542 | 15.620 | 16.361 | 1.549 |

## 2026-06-25 04:56:42 UTC | Ijcz9y7h2Dw_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Ijcz9y7h2Dw_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `82.615` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.694 |
| save_clips | - |
| sample_frames | 0.181 |
| caption_frames | 9.828 |
| sample_fps | 1.543 |
| detect_object_yolo | 4.776 |
| audio_scan | 11.251 |
| asr_timings | 10.478 |
| ast_timings | 4.405 |
| describe_scenes | 4.542 |
| summarize_scenes | 15.620 |
| synthesize_synopsis | 16.361 |
| make_embedding | 1.549 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 10.015 |
| branch_yolo_total | 6.324 |
| branch_audio_total | 26.143 |
