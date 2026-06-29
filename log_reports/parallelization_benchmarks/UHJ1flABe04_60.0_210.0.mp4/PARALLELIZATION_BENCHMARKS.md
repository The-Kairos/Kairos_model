# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 18:51:34 UTC | UHJ1flABe04_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 138.100 | 0.629 | 42.823 | 10.565 | 31.118 | 13.077 | 2.545 |

## 2026-06-25 18:51:34 UTC | UHJ1flABe04_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/UHJ1flABe04_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `138.100` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.629 |
| save_clips | - |
| sample_frames | 0.638 |
| caption_frames | 26.107 |
| sample_fps | 1.864 |
| detect_object_yolo | 7.334 |
| audio_scan | 13.898 |
| asr_timings | 10.146 |
| ast_timings | 18.770 |
| describe_scenes | 10.565 |
| summarize_scenes | 31.118 |
| synthesize_synopsis | 13.077 |
| make_embedding | 2.545 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 26.751 |
| branch_yolo_total | 9.204 |
| branch_audio_total | 42.823 |
