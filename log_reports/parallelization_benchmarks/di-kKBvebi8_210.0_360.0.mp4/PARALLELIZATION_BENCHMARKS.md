# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:16:11 UTC | di-kKBvebi8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 141.795 | 0.788 | 53.363 | 10.901 | 11.986 | 12.377 | 3.312 |

## 2026-06-26 03:16:11 UTC | di-kKBvebi8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/di-kKBvebi8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `141.795` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.788 |
| save_clips | - |
| sample_frames | 1.087 |
| caption_frames | 35.737 |
| sample_fps | 2.259 |
| detect_object_yolo | 8.577 |
| audio_scan | 14.110 |
| asr_timings | 12.115 |
| ast_timings | 27.129 |
| describe_scenes | 10.901 |
| summarize_scenes | 11.986 |
| synthesize_synopsis | 12.377 |
| make_embedding | 3.312 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.829 |
| branch_yolo_total | 10.841 |
| branch_audio_total | 53.363 |
