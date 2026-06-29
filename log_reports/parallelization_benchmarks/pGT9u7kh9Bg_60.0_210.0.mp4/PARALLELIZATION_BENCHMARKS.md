# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 07:49:25 UTC | pGT9u7kh9Bg_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 173.818 | 0.706 | 64.366 | 13.666 | 12.076 | 9.289 | 4.979 |

## 2026-06-28 07:49:25 UTC | pGT9u7kh9Bg_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/pGT9u7kh9Bg_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `173.818` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.706 |
| save_clips | - |
| sample_frames | 1.486 |
| caption_frames | 52.802 |
| sample_fps | 2.343 |
| detect_object_yolo | 10.718 |
| audio_scan | 15.946 |
| asr_timings | 7.640 |
| ast_timings | 40.771 |
| describe_scenes | 13.666 |
| summarize_scenes | 12.076 |
| synthesize_synopsis | 9.289 |
| make_embedding | 4.979 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 54.294 |
| branch_yolo_total | 13.067 |
| branch_audio_total | 64.366 |
