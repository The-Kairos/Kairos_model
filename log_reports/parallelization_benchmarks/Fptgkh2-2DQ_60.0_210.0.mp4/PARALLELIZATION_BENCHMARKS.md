# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 00:38:43 UTC | Fptgkh2-2DQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 150.463 | 0.796 | 51.722 | 10.945 | 13.456 | 23.007 | 3.307 |

## 2026-06-25 00:38:43 UTC | Fptgkh2-2DQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Fptgkh2-2DQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `150.463` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 0.787 |
| caption_frames | 34.588 |
| sample_fps | 2.072 |
| detect_object_yolo | 8.358 |
| audio_scan | 15.025 |
| asr_timings | 10.115 |
| ast_timings | 26.574 |
| describe_scenes | 10.945 |
| summarize_scenes | 13.456 |
| synthesize_synopsis | 23.007 |
| make_embedding | 3.307 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 35.381 |
| branch_yolo_total | 10.436 |
| branch_audio_total | 51.722 |
