# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 10:07:09 UTC | Mulb9_R8n2E_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 203.968 | 0.779 | 79.614 | 29.574 | 13.533 | 19.558 | 3.664 |

## 2026-06-25 10:07:09 UTC | Mulb9_R8n2E_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/Mulb9_R8n2E_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `203.968` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.271 |
| caption_frames | 43.080 |
| sample_fps | 2.364 |
| detect_object_yolo | 9.050 |
| audio_scan | 14.176 |
| asr_timings | 29.424 |
| ast_timings | 36.006 |
| describe_scenes | 29.574 |
| summarize_scenes | 13.533 |
| synthesize_synopsis | 19.558 |
| make_embedding | 3.664 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 44.358 |
| branch_yolo_total | 11.421 |
| branch_audio_total | 79.614 |
