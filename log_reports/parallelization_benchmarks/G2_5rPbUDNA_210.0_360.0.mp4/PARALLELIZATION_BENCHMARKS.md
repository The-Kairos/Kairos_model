# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:09:14 UTC | G2_5rPbUDNA_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 202.823 | 0.632 | 67.479 | 25.896 | 14.025 | 15.153 | 5.386 |

## 2026-06-25 01:09:14 UTC | G2_5rPbUDNA_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/G2_5rPbUDNA_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `202.823` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.632 |
| save_clips | - |
| sample_frames | 1.194 |
| caption_frames | 58.356 |
| sample_fps | 2.262 |
| detect_object_yolo | 11.054 |
| audio_scan | 14.887 |
| asr_timings | 9.383 |
| ast_timings | 43.200 |
| describe_scenes | 25.896 |
| summarize_scenes | 14.025 |
| synthesize_synopsis | 15.153 |
| make_embedding | 5.386 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 59.556 |
| branch_yolo_total | 13.321 |
| branch_audio_total | 67.479 |
