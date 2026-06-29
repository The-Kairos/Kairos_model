# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 03:48:57 UTC | xOJWLSOp3zU_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 131.544 | 0.781 | 53.214 | 10.851 | 7.337 | 6.362 | 3.300 |

## 2026-06-27 03:48:57 UTC | xOJWLSOp3zU_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/xOJWLSOp3zU_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `131.544` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.781 |
| save_clips | - |
| sample_frames | 1.083 |
| caption_frames | 36.662 |
| sample_fps | 2.290 |
| detect_object_yolo | 8.252 |
| audio_scan | 16.266 |
| asr_timings | 10.311 |
| ast_timings | 26.628 |
| describe_scenes | 10.851 |
| summarize_scenes | 7.337 |
| synthesize_synopsis | 6.362 |
| make_embedding | 3.300 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 37.751 |
| branch_yolo_total | 10.548 |
| branch_audio_total | 53.214 |
