# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 08:12:25 UTC | iCbjk3uRtqA_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 213.330 | 0.779 | 64.040 | 33.288 | 17.665 | 27.344 | 4.582 |

## 2026-06-26 08:12:25 UTC | iCbjk3uRtqA_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/iCbjk3uRtqA_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `213.330` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.779 |
| save_clips | - |
| sample_frames | 1.350 |
| caption_frames | 50.291 |
| sample_fps | 2.447 |
| detect_object_yolo | 10.131 |
| audio_scan | 15.167 |
| asr_timings | 10.592 |
| ast_timings | 38.273 |
| describe_scenes | 33.288 |
| summarize_scenes | 17.665 |
| synthesize_synopsis | 27.344 |
| make_embedding | 4.582 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 51.647 |
| branch_yolo_total | 12.583 |
| branch_audio_total | 64.040 |
