# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:35:03 UTC | lyGaTk4MLVM_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 181.278 | 0.742 | 55.988 | 18.352 | 13.300 | 27.907 | 4.226 |

## 2026-06-26 17:35:03 UTC | lyGaTk4MLVM_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lyGaTk4MLVM_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `181.278` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.742 |
| save_clips | - |
| sample_frames | 1.026 |
| caption_frames | 46.257 |
| sample_fps | 2.186 |
| detect_object_yolo | 9.854 |
| audio_scan | 12.885 |
| asr_timings | 7.814 |
| ast_timings | 35.281 |
| describe_scenes | 18.352 |
| summarize_scenes | 13.300 |
| synthesize_synopsis | 27.907 |
| make_embedding | 4.226 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.289 |
| branch_yolo_total | 12.046 |
| branch_audio_total | 55.988 |
