# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 13:46:55 UTC | PPSICA2UeP0_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.215 | 0.798 | 43.657 | 15.444 | 11.867 | 46.914 | 2.796 |

## 2026-06-25 13:46:55 UTC | PPSICA2UeP0_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/PPSICA2UeP0_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.215` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.798 |
| save_clips | - |
| sample_frames | 0.700 |
| caption_frames | 31.534 |
| sample_fps | 2.079 |
| detect_object_yolo | 7.952 |
| audio_scan | 12.177 |
| asr_timings | 9.994 |
| ast_timings | 21.478 |
| describe_scenes | 15.444 |
| summarize_scenes | 11.867 |
| synthesize_synopsis | 46.914 |
| make_embedding | 2.796 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.240 |
| branch_yolo_total | 10.038 |
| branch_audio_total | 43.657 |
