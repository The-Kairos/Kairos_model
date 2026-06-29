# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:17:18 UTC | G2_5rPbUDNA_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 209.308 | 0.638 | 70.169 | 22.554 | 14.657 | 17.827 | 5.694 |

## 2026-06-25 01:17:18 UTC | G2_5rPbUDNA_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G2_5rPbUDNA_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `209.308` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.638 |
| save_clips | - |
| sample_frames | 1.357 |
| caption_frames | 60.487 |
| sample_fps | 2.345 |
| detect_object_yolo | 12.113 |
| audio_scan | 12.952 |
| asr_timings | 11.036 |
| ast_timings | 46.172 |
| describe_scenes | 22.554 |
| summarize_scenes | 14.657 |
| synthesize_synopsis | 17.827 |
| make_embedding | 5.694 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 61.850 |
| branch_yolo_total | 14.465 |
| branch_audio_total | 70.169 |
