# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 13:25:21 UTC | kBAWeVHNlBo_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 236.137 | 0.818 | 69.962 | 22.959 | 35.276 | 25.429 | 5.811 |

## 2026-06-26 13:25:21 UTC | kBAWeVHNlBo_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/kBAWeVHNlBo_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `236.137` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.818 |
| save_clips | - |
| sample_frames | 1.347 |
| caption_frames | 59.268 |
| sample_fps | 2.502 |
| detect_object_yolo | 11.335 |
| audio_scan | 16.192 |
| asr_timings | 9.726 |
| ast_timings | 44.035 |
| describe_scenes | 22.959 |
| summarize_scenes | 35.276 |
| synthesize_synopsis | 25.429 |
| make_embedding | 5.811 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 60.620 |
| branch_yolo_total | 13.844 |
| branch_audio_total | 69.962 |
