# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:43:43 UTC | fuf0Ma1Ozc8_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 116.807 | 0.803 | 30.743 | 11.101 | 25.727 | 9.742 | 2.774 |

## 2026-06-26 04:43:43 UTC | fuf0Ma1Ozc8_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fuf0Ma1Ozc8_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `116.807` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 0.980 |
| caption_frames | 29.757 |
| sample_fps | 2.147 |
| detect_object_yolo | 7.126 |
| audio_scan | 3.884 |
| asr_timings | 0.000 |
| ast_timings | 21.370 |
| describe_scenes | 11.101 |
| summarize_scenes | 25.727 |
| synthesize_synopsis | 9.742 |
| make_embedding | 2.774 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 30.743 |
| branch_yolo_total | 9.278 |
| branch_audio_total | 25.262 |
