# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:24:59 UTC | IDvx9c2f_VY_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 187.426 | 0.715 | 68.676 | 16.544 | 12.116 | 10.821 | 5.591 |

## 2026-06-25 04:24:59 UTC | IDvx9c2f_VY_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IDvx9c2f_VY_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `187.426` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.715 |
| save_clips | - |
| sample_frames | 1.759 |
| caption_frames | 56.019 |
| sample_fps | 2.538 |
| detect_object_yolo | 11.247 |
| audio_scan | 14.946 |
| asr_timings | 9.931 |
| ast_timings | 43.790 |
| describe_scenes | 16.544 |
| summarize_scenes | 12.116 |
| synthesize_synopsis | 10.821 |
| make_embedding | 5.591 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.784 |
| branch_yolo_total | 13.790 |
| branch_audio_total | 68.676 |
