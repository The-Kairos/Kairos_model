# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 15:12:59 UTC | lLepYM0nQhc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 164.309 | 0.793 | 67.076 | 13.634 | 11.545 | 22.192 | 3.062 |

## 2026-06-26 15:12:59 UTC | lLepYM0nQhc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lLepYM0nQhc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `164.309` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.793 |
| save_clips | - |
| sample_frames | 0.861 |
| caption_frames | 33.446 |
| sample_fps | 2.191 |
| detect_object_yolo | 8.111 |
| audio_scan | 15.118 |
| asr_timings | 27.110 |
| ast_timings | 24.839 |
| describe_scenes | 13.634 |
| summarize_scenes | 11.545 |
| synthesize_synopsis | 22.192 |
| make_embedding | 3.062 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 34.314 |
| branch_yolo_total | 10.308 |
| branch_audio_total | 67.076 |
