# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 03:24:37 UTC | di-kKBvebi8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.966 | 0.793 | 63.054 | 13.931 | 9.340 | 13.388 | 4.174 |

## 2026-06-26 03:24:37 UTC | di-kKBvebi8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/di-kKBvebi8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.966` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.605 |
| caption_frames | 46.035 |
| sample_fps | 2.447 |
| detect_object_yolo | 9.804 |
| audio_scan | 16.282 |
| asr_timings | 11.779 |
| ast_timings | 34.985 |
| describe_scenes | 13.931 |
| summarize_scenes | 9.340 |
| synthesize_synopsis | 13.388 |
| make_embedding | 4.174 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.646 |
| branch_yolo_total | 12.257 |
| branch_audio_total | 63.054 |
