# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:13:00 UTC | 8sWfdCTFW8o_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.932 | 0.687 | 52.143 | 15.597 | 29.130 | 15.583 | 3.334 |

## 2026-06-24 17:13:00 UTC | 8sWfdCTFW8o_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8sWfdCTFW8o_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.932` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.687 |
| save_clips | - |
| sample_frames | 1.167 |
| caption_frames | 35.058 |
| sample_fps | 2.094 |
| detect_object_yolo | 8.718 |
| audio_scan | 16.023 |
| asr_timings | 9.375 |
| ast_timings | 26.737 |
| describe_scenes | 15.597 |
| summarize_scenes | 29.130 |
| synthesize_synopsis | 15.583 |
| make_embedding | 3.334 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 36.231 |
| branch_yolo_total | 10.817 |
| branch_audio_total | 52.143 |
