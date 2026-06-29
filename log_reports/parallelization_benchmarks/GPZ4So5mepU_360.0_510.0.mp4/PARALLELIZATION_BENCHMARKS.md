# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:48:37 UTC | GPZ4So5mepU_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 115.803 | 0.672 | 46.278 | 7.851 | 6.855 | 11.254 | 2.787 |

## 2026-06-25 01:48:37 UTC | GPZ4So5mepU_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GPZ4So5mepU_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `115.803` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.672 |
| save_clips | - |
| sample_frames | 0.708 |
| caption_frames | 28.402 |
| sample_fps | 1.964 |
| detect_object_yolo | 7.551 |
| audio_scan | 16.128 |
| asr_timings | 8.962 |
| ast_timings | 21.179 |
| describe_scenes | 7.851 |
| summarize_scenes | 6.855 |
| synthesize_synopsis | 11.254 |
| make_embedding | 2.787 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 29.115 |
| branch_yolo_total | 9.521 |
| branch_audio_total | 46.278 |
