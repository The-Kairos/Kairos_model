# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:14:16 UTC | AV9a_rRCOaU_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 166.615 | 0.787 | 56.036 | 15.712 | 21.397 | 13.128 | 3.883 |

## 2026-06-24 19:14:16 UTC | AV9a_rRCOaU_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AV9a_rRCOaU_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `166.615` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.787 |
| save_clips | - |
| sample_frames | 1.218 |
| caption_frames | 41.685 |
| sample_fps | 2.311 |
| detect_object_yolo | 9.051 |
| audio_scan | 14.944 |
| asr_timings | 9.233 |
| ast_timings | 31.851 |
| describe_scenes | 15.712 |
| summarize_scenes | 21.397 |
| synthesize_synopsis | 13.128 |
| make_embedding | 3.883 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.909 |
| branch_yolo_total | 11.368 |
| branch_audio_total | 56.036 |
