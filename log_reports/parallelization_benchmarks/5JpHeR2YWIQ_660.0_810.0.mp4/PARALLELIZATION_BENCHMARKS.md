# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 11:51:56 UTC | 5JpHeR2YWIQ_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 114.051 | 0.718 | 34.271 | 10.067 | 10.541 | 22.063 | 2.280 |

## 2026-06-24 11:51:56 UTC | 5JpHeR2YWIQ_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/5JpHeR2YWIQ_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `114.051` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.718 |
| save_clips | - |
| sample_frames | 0.656 |
| caption_frames | 23.932 |
| sample_fps | 1.807 |
| detect_object_yolo | 6.326 |
| audio_scan | 8.121 |
| asr_timings | 10.662 |
| ast_timings | 15.480 |
| describe_scenes | 10.067 |
| summarize_scenes | 10.541 |
| synthesize_synopsis | 22.063 |
| make_embedding | 2.280 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.594 |
| branch_yolo_total | 8.138 |
| branch_audio_total | 34.271 |
