# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:41:46 UTC | fuf0Ma1Ozc8_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 107.634 | 0.791 | 34.339 | 12.969 | 6.089 | 11.779 | 3.070 |

## 2026-06-26 04:41:46 UTC | fuf0Ma1Ozc8_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fuf0Ma1Ozc8_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `107.634` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.791 |
| save_clips | - |
| sample_frames | 1.111 |
| caption_frames | 33.222 |
| sample_fps | 2.238 |
| detect_object_yolo | 7.455 |
| audio_scan | 3.868 |
| asr_timings | 0.000 |
| ast_timings | 23.624 |
| describe_scenes | 12.969 |
| summarize_scenes | 6.089 |
| synthesize_synopsis | 11.779 |
| make_embedding | 3.070 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.339 |
| branch_yolo_total | 9.699 |
| branch_audio_total | 27.500 |
