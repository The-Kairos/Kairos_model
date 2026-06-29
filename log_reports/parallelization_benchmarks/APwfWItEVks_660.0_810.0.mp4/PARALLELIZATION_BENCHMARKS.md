# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 18:56:56 UTC | APwfWItEVks_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 136.348 | 0.772 | 47.216 | 9.552 | 18.769 | 13.017 | 3.064 |

## 2026-06-24 18:56:56 UTC | APwfWItEVks_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/APwfWItEVks_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `136.348` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.772 |
| save_clips | - |
| sample_frames | 0.738 |
| caption_frames | 31.828 |
| sample_fps | 2.114 |
| detect_object_yolo | 7.888 |
| audio_scan | 13.836 |
| asr_timings | 9.222 |
| ast_timings | 24.151 |
| describe_scenes | 9.552 |
| summarize_scenes | 18.769 |
| synthesize_synopsis | 13.017 |
| make_embedding | 3.064 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 32.572 |
| branch_yolo_total | 10.008 |
| branch_audio_total | 47.216 |
