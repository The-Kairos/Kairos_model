# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 23:51:00 UTC | _YfQuBuAmHs_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 145.795 | 0.716 | 60.465 | 11.524 | 7.444 | 9.823 | 3.550 |

## 2026-06-25 23:51:00 UTC | _YfQuBuAmHs_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/_YfQuBuAmHs_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `145.795` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.716 |
| save_clips | - |
| sample_frames | 1.017 |
| caption_frames | 39.308 |
| sample_fps | 2.114 |
| detect_object_yolo | 8.405 |
| audio_scan | 13.927 |
| asr_timings | 15.798 |
| ast_timings | 30.732 |
| describe_scenes | 11.524 |
| summarize_scenes | 7.444 |
| synthesize_synopsis | 9.823 |
| make_embedding | 3.550 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 40.331 |
| branch_yolo_total | 10.524 |
| branch_audio_total | 60.465 |
