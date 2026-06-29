# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-28 09:16:31 UTC | r5er3RvgtBc_210.0_360.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.685 | 0.789 | 55.874 | 13.179 | 8.126 | 9.273 | 4.117 |

## 2026-06-28 09:16:31 UTC | r5er3RvgtBc_210.0_360.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/r5er3RvgtBc_210.0_360.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.685` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.789 |
| save_clips | - |
| sample_frames | 1.407 |
| caption_frames | 45.633 |
| sample_fps | 2.332 |
| detect_object_yolo | 9.561 |
| audio_scan | 9.624 |
| asr_timings | 11.057 |
| ast_timings | 35.184 |
| describe_scenes | 13.179 |
| summarize_scenes | 8.126 |
| synthesize_synopsis | 9.273 |
| make_embedding | 4.117 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.046 |
| branch_yolo_total | 11.899 |
| branch_audio_total | 55.874 |
