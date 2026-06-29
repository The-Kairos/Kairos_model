# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 01:58:26 UTC | c8EAPVV4dVc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 165.879 | 0.796 | 59.090 | 15.403 | 10.197 | 11.384 | 4.475 |

## 2026-06-26 01:58:26 UTC | c8EAPVV4dVc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/c8EAPVV4dVc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `165.879` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.796 |
| save_clips | - |
| sample_frames | 1.180 |
| caption_frames | 49.390 |
| sample_fps | 2.319 |
| detect_object_yolo | 10.177 |
| audio_scan | 8.778 |
| asr_timings | 11.074 |
| ast_timings | 39.230 |
| describe_scenes | 15.403 |
| summarize_scenes | 10.197 |
| synthesize_synopsis | 11.384 |
| make_embedding | 4.475 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 50.576 |
| branch_yolo_total | 12.502 |
| branch_audio_total | 59.090 |
