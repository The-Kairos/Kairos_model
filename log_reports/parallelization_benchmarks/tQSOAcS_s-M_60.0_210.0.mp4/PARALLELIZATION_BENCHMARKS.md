# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 23:37:34 UTC | tQSOAcS_s-M_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 215.496 | 0.827 | 76.142 | 19.125 | 13.008 | 13.065 | 6.370 |

## 2026-06-26 23:37:34 UTC | tQSOAcS_s-M_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/tQSOAcS_s-M_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `215.496` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.827 |
| save_clips | - |
| sample_frames | 1.569 |
| caption_frames | 68.542 |
| sample_fps | 2.558 |
| detect_object_yolo | 12.853 |
| audio_scan | 12.870 |
| asr_timings | 11.487 |
| ast_timings | 51.776 |
| describe_scenes | 19.125 |
| summarize_scenes | 13.008 |
| synthesize_synopsis | 13.065 |
| make_embedding | 6.370 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 70.117 |
| branch_yolo_total | 15.416 |
| branch_audio_total | 76.142 |
