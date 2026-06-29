# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:05:46 UTC | bUa-0ptWL5M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 98.583 | 0.773 | 39.265 | 6.728 | 5.709 | 9.256 | 2.339 |

## 2026-06-26 01:05:46 UTC | bUa-0ptWL5M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/bUa-0ptWL5M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `98.583` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.773 |
| save_clips | - |
| sample_frames | 0.495 |
| caption_frames | 23.697 |
| sample_fps | 1.967 |
| detect_object_yolo | 6.970 |
| audio_scan | 13.746 |
| asr_timings | 9.664 |
| ast_timings | 15.845 |
| describe_scenes | 6.728 |
| summarize_scenes | 5.709 |
| synthesize_synopsis | 9.256 |
| make_embedding | 2.339 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.197 |
| branch_yolo_total | 8.943 |
| branch_audio_total | 39.265 |
