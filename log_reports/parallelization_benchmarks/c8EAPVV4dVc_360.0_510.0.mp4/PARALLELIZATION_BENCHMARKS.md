# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 02:00:54 UTC | c8EAPVV4dVc_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 146.603 | 0.823 | 51.354 | 13.165 | 12.274 | 10.672 | 3.612 |

## 2026-06-26 02:00:54 UTC | c8EAPVV4dVc_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/c8EAPVV4dVc_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `146.603` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.823 |
| save_clips | - |
| sample_frames | 1.062 |
| caption_frames | 40.905 |
| sample_fps | 2.268 |
| detect_object_yolo | 9.073 |
| audio_scan | 11.782 |
| asr_timings | 9.062 |
| ast_timings | 30.490 |
| describe_scenes | 13.165 |
| summarize_scenes | 12.274 |
| synthesize_synopsis | 10.672 |
| make_embedding | 3.612 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.973 |
| branch_yolo_total | 11.347 |
| branch_audio_total | 51.354 |
