# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-04-11 17:22:28 UTC | K-fold_Cross_Validation.mp4 | parallel | gemini | gemini-embedding-001 | 188.860 | 8.612 | 122.869 | 32.462 | 10.667 | 6.536 | 2.731 |

## 2026-04-11 17:22:28 UTC | K-fold_Cross_Validation.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/72e5e374-aa34-4b80-ad3b-c3062ee42677/K-fold_Cross_Validation.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.860` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 8.612 |
| save_clips | - |
| sample_frames | 2.133 |
| caption_frames | 58.957 |
| sample_fps | 10.709 |
| detect_object_yolo | 28.719 |
| audio_scan | 47.565 |
| asr_timings | 39.324 |
| ast_timings | 75.294 |
| describe_scenes | 32.462 |
| summarize_scenes | 10.667 |
| synthesize_synopsis | 6.536 |
| make_embedding | 2.731 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.098 |
| branch_yolo_total | 39.436 |
| branch_audio_total | 122.869 |
