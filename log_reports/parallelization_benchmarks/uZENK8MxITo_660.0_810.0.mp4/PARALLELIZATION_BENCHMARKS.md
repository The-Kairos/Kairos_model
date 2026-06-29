# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:23:51 UTC | uZENK8MxITo_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 151.277 | 0.774 | 95.235 | 8.793 | 4.355 | 7.043 | 2.255 |

## 2026-06-27 01:23:51 UTC | uZENK8MxITo_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uZENK8MxITo_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `151.277` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.774 |
| save_clips | - |
| sample_frames | 0.624 |
| caption_frames | 21.516 |
| sample_fps | 2.002 |
| detect_object_yolo | 7.283 |
| audio_scan | 15.956 |
| asr_timings | 63.417 |
| ast_timings | 15.852 |
| describe_scenes | 8.793 |
| summarize_scenes | 4.355 |
| synthesize_synopsis | 7.043 |
| make_embedding | 2.255 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 22.146 |
| branch_yolo_total | 9.290 |
| branch_audio_total | 95.235 |
