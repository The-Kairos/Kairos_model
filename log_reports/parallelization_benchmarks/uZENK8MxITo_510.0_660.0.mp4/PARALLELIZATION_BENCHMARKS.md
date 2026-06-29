# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-27 01:19:46 UTC | uZENK8MxITo_510.0_660.0.mp4 | sequential | gemini | gemini-embedding-001 | 127.959 | 0.761 | 77.952 | 4.583 | 3.786 | 8.380 | 2.035 |

## 2026-06-27 01:19:46 UTC | uZENK8MxITo_510.0_660.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/uZENK8MxITo_510.0_660.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `127.959` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.761 |
| save_clips | - |
| sample_frames | 0.393 |
| caption_frames | 19.544 |
| sample_fps | 1.932 |
| detect_object_yolo | 7.201 |
| audio_scan | 13.933 |
| asr_timings | 50.964 |
| ast_timings | 13.046 |
| describe_scenes | 4.583 |
| summarize_scenes | 3.786 |
| synthesize_synopsis | 8.380 |
| make_embedding | 2.035 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 19.943 |
| branch_yolo_total | 9.139 |
| branch_audio_total | 77.952 |
