# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 12:09:58 UTC | jqt8j8h_U_8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 192.556 | 0.674 | 54.120 | 19.645 | 20.599 | 32.454 | 4.201 |

## 2026-06-26 12:09:58 UTC | jqt8j8h_U_8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/jqt8j8h_U_8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `192.556` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.674 |
| save_clips | - |
| sample_frames | 1.521 |
| caption_frames | 45.836 |
| sample_fps | 2.324 |
| detect_object_yolo | 9.775 |
| audio_scan | 8.644 |
| asr_timings | 10.024 |
| ast_timings | 35.443 |
| describe_scenes | 19.645 |
| summarize_scenes | 20.599 |
| synthesize_synopsis | 32.454 |
| make_embedding | 4.201 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.363 |
| branch_yolo_total | 12.104 |
| branch_audio_total | 54.120 |
