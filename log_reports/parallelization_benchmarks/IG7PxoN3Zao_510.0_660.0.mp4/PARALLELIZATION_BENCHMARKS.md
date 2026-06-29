# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 04:33:50 UTC | IG7PxoN3Zao_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 177.317 | 0.663 | 56.778 | 16.151 | 16.544 | 16.452 | 4.684 |

## 2026-06-25 04:33:50 UTC | IG7PxoN3Zao_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/IG7PxoN3Zao_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `177.317` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.663 |
| save_clips | - |
| sample_frames | 1.293 |
| caption_frames | 50.132 |
| sample_fps | 2.332 |
| detect_object_yolo | 10.827 |
| audio_scan | 7.564 |
| asr_timings | 10.453 |
| ast_timings | 38.753 |
| describe_scenes | 16.151 |
| summarize_scenes | 16.544 |
| synthesize_synopsis | 16.452 |
| make_embedding | 4.684 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.431 |
| branch_yolo_total | 13.165 |
| branch_audio_total | 56.778 |
