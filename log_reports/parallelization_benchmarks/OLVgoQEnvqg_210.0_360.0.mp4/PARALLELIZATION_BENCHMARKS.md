# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 11:37:51 UTC | OLVgoQEnvqg_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 171.547 | 0.803 | 54.312 | 16.065 | 16.523 | 17.965 | 4.149 |

## 2026-06-25 11:37:51 UTC | OLVgoQEnvqg_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/OLVgoQEnvqg_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `171.547` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.803 |
| save_clips | - |
| sample_frames | 1.293 |
| caption_frames | 46.540 |
| sample_fps | 2.347 |
| detect_object_yolo | 10.127 |
| audio_scan | 8.924 |
| asr_timings | 9.720 |
| ast_timings | 35.660 |
| describe_scenes | 16.065 |
| summarize_scenes | 16.523 |
| synthesize_synopsis | 17.965 |
| make_embedding | 4.149 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.839 |
| branch_yolo_total | 12.480 |
| branch_audio_total | 54.312 |
