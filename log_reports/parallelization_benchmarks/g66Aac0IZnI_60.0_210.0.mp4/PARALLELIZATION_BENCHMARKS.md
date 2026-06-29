# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:57:48 UTC | g66Aac0IZnI_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 159.801 | 0.836 | 51.445 | 12.293 | 8.425 | 19.020 | 4.182 |

## 2026-06-26 04:57:48 UTC | g66Aac0IZnI_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/g66Aac0IZnI_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `159.801` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.836 |
| save_clips | - |
| sample_frames | 1.387 |
| caption_frames | 48.007 |
| sample_fps | 2.508 |
| detect_object_yolo | 10.224 |
| audio_scan | 6.603 |
| asr_timings | 8.557 |
| ast_timings | 36.277 |
| describe_scenes | 12.293 |
| summarize_scenes | 8.425 |
| synthesize_synopsis | 19.020 |
| make_embedding | 4.182 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.400 |
| branch_yolo_total | 12.739 |
| branch_audio_total | 51.445 |
