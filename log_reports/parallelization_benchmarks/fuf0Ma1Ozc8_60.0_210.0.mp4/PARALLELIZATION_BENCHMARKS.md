# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 04:46:09 UTC | fuf0Ma1Ozc8_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 144.363 | 0.793 | 47.240 | 17.699 | 9.395 | 13.437 | 4.110 |

## 2026-06-26 04:46:09 UTC | fuf0Ma1Ozc8_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/fuf0Ma1Ozc8_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `144.363` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 1.685 |
| caption_frames | 45.549 |
| sample_fps | 2.490 |
| detect_object_yolo | 9.144 |
| audio_scan | 3.922 |
| asr_timings | 0.000 |
| ast_timings | 34.710 |
| describe_scenes | 17.699 |
| summarize_scenes | 9.395 |
| synthesize_synopsis | 13.437 |
| make_embedding | 4.110 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.240 |
| branch_yolo_total | 11.640 |
| branch_audio_total | 38.641 |
