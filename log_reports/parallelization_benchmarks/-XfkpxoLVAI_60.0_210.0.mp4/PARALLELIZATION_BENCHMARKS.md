# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:35:36 UTC | -XfkpxoLVAI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 140.504 | 0.703 | 45.377 | 15.024 | 12.279 | 30.047 | 2.551 |

## 2026-06-24 08:35:36 UTC | -XfkpxoLVAI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-XfkpxoLVAI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `140.504` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.703 |
| save_clips | - |
| sample_frames | 0.716 |
| caption_frames | 24.033 |
| sample_fps | 1.937 |
| detect_object_yolo | 6.509 |
| audio_scan | 9.649 |
| asr_timings | 17.199 |
| ast_timings | 18.520 |
| describe_scenes | 15.024 |
| summarize_scenes | 12.279 |
| synthesize_synopsis | 30.047 |
| make_embedding | 2.551 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.756 |
| branch_yolo_total | 8.452 |
| branch_audio_total | 45.377 |
