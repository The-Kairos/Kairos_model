# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:30:52 UTC | IG7PxoN3Zao_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 179.632 | 0.750 | 58.688 | 13.959 | 12.256 | 20.639 | 5.056 |

## 2026-06-25 04:30:52 UTC | IG7PxoN3Zao_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IG7PxoN3Zao_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `179.632` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.750 |
| save_clips | - |
| sample_frames | 1.306 |
| caption_frames | 52.500 |
| sample_fps | 2.326 |
| detect_object_yolo | 10.751 |
| audio_scan | 6.461 |
| asr_timings | 11.277 |
| ast_timings | 40.942 |
| describe_scenes | 13.959 |
| summarize_scenes | 12.256 |
| synthesize_synopsis | 20.639 |
| make_embedding | 5.056 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 53.812 |
| branch_yolo_total | 13.083 |
| branch_audio_total | 58.688 |
