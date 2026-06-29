# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-13 12:07:23 UTC | Ready_Player_1.mp4 | parallel | gemini | gemini-embedding-001 | 199.211 | 1.421 | 83.467 | 61.562 | 29.612 | 15.681 | 2.384 |

## 2026-05-13 12:07:23 UTC | Ready_Player_1.mp4 | parallel

- Video path: `/home/Kairos_model/.tmp/kairos/jobs/07be7511-48a7-448a-ab1e-ea2f50b7e445/Ready_Player_1.mp4`
- Low memory mode: `False`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `199.211` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 1.421 |
| save_clips | - |
| sample_frames | 1.692 |
| caption_frames | 40.245 |
| sample_fps | 4.692 |
| detect_object_yolo | 16.127 |
| audio_scan | 29.693 |
| asr_timings | 12.788 |
| ast_timings | 53.763 |
| describe_scenes | 61.562 |
| summarize_scenes | 29.612 |
| synthesize_synopsis | 15.681 |
| make_embedding | 2.384 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.944 |
| branch_yolo_total | 20.832 |
| branch_audio_total | 83.467 |
