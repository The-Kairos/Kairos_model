# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 12:35:48 UTC | 6g0IdjnX9fQ_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 105.991 | 0.733 | 33.019 | 14.868 | 6.428 | 20.592 | 2.045 |

## 2026-06-24 12:35:48 UTC | 6g0IdjnX9fQ_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/6g0IdjnX9fQ_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `105.991` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.733 |
| save_clips | - |
| sample_frames | 0.388 |
| caption_frames | 18.260 |
| sample_fps | 1.782 |
| detect_object_yolo | 6.479 |
| audio_scan | 11.707 |
| asr_timings | 8.409 |
| ast_timings | 12.895 |
| describe_scenes | 14.868 |
| summarize_scenes | 6.428 |
| synthesize_synopsis | 20.592 |
| make_embedding | 2.045 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 18.654 |
| branch_yolo_total | 8.267 |
| branch_audio_total | 33.019 |
