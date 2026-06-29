# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 05:58:55 UTC | K8o5XoeNjC0_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 201.901 | 0.646 | 61.625 | 20.575 | 26.737 | 14.940 | 5.052 |

## 2026-06-25 05:58:55 UTC | K8o5XoeNjC0_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/K8o5XoeNjC0_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `201.901` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.646 |
| save_clips | - |
| sample_frames | 1.432 |
| caption_frames | 56.254 |
| sample_fps | 2.325 |
| detect_object_yolo | 10.916 |
| audio_scan | 12.721 |
| asr_timings | 7.741 |
| ast_timings | 41.154 |
| describe_scenes | 20.575 |
| summarize_scenes | 26.737 |
| synthesize_synopsis | 14.940 |
| make_embedding | 5.052 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 57.692 |
| branch_yolo_total | 13.247 |
| branch_audio_total | 61.625 |
