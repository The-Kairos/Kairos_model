# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:59:50 UTC | pKGH1tyyCcY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.123 | 0.780 | 86.078 | 13.694 | 8.026 | 6.948 | 4.984 |

## 2026-06-28 07:59:50 UTC | pKGH1tyyCcY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pKGH1tyyCcY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.123` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.780 |
| save_clips | - |
| sample_frames | 1.403 |
| caption_frames | 51.990 |
| sample_fps | 2.451 |
| detect_object_yolo | 10.397 |
| audio_scan | 8.557 |
| asr_timings | 37.319 |
| ast_timings | 40.194 |
| describe_scenes | 13.694 |
| summarize_scenes | 8.026 |
| synthesize_synopsis | 6.948 |
| make_embedding | 4.984 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.399 |
| branch_yolo_total | 12.854 |
| branch_audio_total | 86.078 |
