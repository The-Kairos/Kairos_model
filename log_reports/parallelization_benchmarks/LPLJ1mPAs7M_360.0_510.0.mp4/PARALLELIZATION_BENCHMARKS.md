# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 07:17:48 UTC | LPLJ1mPAs7M_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 188.993 | 0.643 | 57.783 | 26.958 | 12.584 | 17.423 | 4.450 |

## 2026-06-25 07:17:48 UTC | LPLJ1mPAs7M_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/LPLJ1mPAs7M_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `188.993` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.643 |
| save_clips | - |
| sample_frames | 1.231 |
| caption_frames | 53.804 |
| sample_fps | 2.298 |
| detect_object_yolo | 10.353 |
| audio_scan | 8.602 |
| asr_timings | 11.624 |
| ast_timings | 37.548 |
| describe_scenes | 26.958 |
| summarize_scenes | 12.584 |
| synthesize_synopsis | 17.423 |
| make_embedding | 4.450 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 55.041 |
| branch_yolo_total | 12.657 |
| branch_audio_total | 57.783 |
