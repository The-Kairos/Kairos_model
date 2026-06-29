# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:33:07 UTC | ccMqbhacbpY_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 146.746 | 0.819 | 48.487 | 8.081 | 20.653 | 19.153 | 3.087 |

## 2026-06-26 02:33:07 UTC | ccMqbhacbpY_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/ccMqbhacbpY_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `146.746` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.819 |
| save_clips | - |
| sample_frames | 1.106 |
| caption_frames | 33.417 |
| sample_fps | 2.173 |
| detect_object_yolo | 8.351 |
| audio_scan | 14.085 |
| asr_timings | 9.827 |
| ast_timings | 24.567 |
| describe_scenes | 8.081 |
| summarize_scenes | 20.653 |
| synthesize_synopsis | 19.153 |
| make_embedding | 3.087 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.529 |
| branch_yolo_total | 10.529 |
| branch_audio_total | 48.487 |
