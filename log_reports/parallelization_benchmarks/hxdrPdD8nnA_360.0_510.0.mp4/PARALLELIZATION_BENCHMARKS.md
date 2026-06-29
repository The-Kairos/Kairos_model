# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 07:25:09 UTC | hxdrPdD8nnA_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 170.467 | 0.840 | 55.969 | 19.743 | 10.163 | 20.044 | 4.095 |

## 2026-06-26 07:25:09 UTC | hxdrPdD8nnA_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/hxdrPdD8nnA_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `170.467` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.840 |
| save_clips | - |
| sample_frames | 1.150 |
| caption_frames | 44.975 |
| sample_fps | 2.398 |
| detect_object_yolo | 9.668 |
| audio_scan | 14.964 |
| asr_timings | 8.734 |
| ast_timings | 32.263 |
| describe_scenes | 19.743 |
| summarize_scenes | 10.163 |
| synthesize_synopsis | 20.044 |
| make_embedding | 4.095 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 46.131 |
| branch_yolo_total | 12.072 |
| branch_audio_total | 55.969 |
