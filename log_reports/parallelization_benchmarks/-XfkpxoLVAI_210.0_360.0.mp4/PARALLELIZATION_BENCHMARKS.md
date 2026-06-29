# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 08:33:15 UTC | -XfkpxoLVAI_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 255.276 | 0.738 | 95.236 | 27.646 | 17.335 | 46.643 | 4.310 |

## 2026-06-24 08:33:15 UTC | -XfkpxoLVAI_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/-XfkpxoLVAI_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `255.276` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.738 |
| save_clips | - |
| sample_frames | 1.866 |
| caption_frames | 48.239 |
| sample_fps | 2.333 |
| detect_object_yolo | 9.532 |
| audio_scan | 9.607 |
| asr_timings | 50.088 |
| ast_timings | 35.533 |
| describe_scenes | 27.646 |
| summarize_scenes | 17.335 |
| synthesize_synopsis | 46.643 |
| make_embedding | 4.310 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.112 |
| branch_yolo_total | 11.870 |
| branch_audio_total | 95.236 |
