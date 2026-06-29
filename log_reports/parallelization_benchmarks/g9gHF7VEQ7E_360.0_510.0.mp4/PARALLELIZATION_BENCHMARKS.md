# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 05:04:14 UTC | g9gHF7VEQ7E_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 196.918 | 0.696 | 67.575 | 14.640 | 18.977 | 18.516 | 5.118 |

## 2026-06-26 05:04:14 UTC | g9gHF7VEQ7E_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/g9gHF7VEQ7E_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `196.918` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.696 |
| save_clips | - |
| sample_frames | 1.932 |
| caption_frames | 54.946 |
| sample_fps | 2.617 |
| detect_object_yolo | 10.493 |
| audio_scan | 15.147 |
| asr_timings | 10.975 |
| ast_timings | 41.444 |
| describe_scenes | 14.640 |
| summarize_scenes | 18.977 |
| synthesize_synopsis | 18.516 |
| make_embedding | 5.118 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 56.885 |
| branch_yolo_total | 13.116 |
| branch_audio_total | 67.575 |
