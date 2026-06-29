# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 09:26:13 UTC | MOThH7E8fzc_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 214.425 | 0.680 | 65.445 | 24.989 | 18.348 | 36.213 | 3.916 |

## 2026-06-25 09:26:13 UTC | MOThH7E8fzc_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/MOThH7E8fzc_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `214.425` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.680 |
| save_clips | - |
| sample_frames | 1.600 |
| caption_frames | 49.324 |
| sample_fps | 2.419 |
| detect_object_yolo | 10.044 |
| audio_scan | 16.085 |
| asr_timings | 16.807 |
| ast_timings | 32.545 |
| describe_scenes | 24.989 |
| summarize_scenes | 18.348 |
| synthesize_synopsis | 36.213 |
| make_embedding | 3.916 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.931 |
| branch_yolo_total | 12.469 |
| branch_audio_total | 65.445 |
