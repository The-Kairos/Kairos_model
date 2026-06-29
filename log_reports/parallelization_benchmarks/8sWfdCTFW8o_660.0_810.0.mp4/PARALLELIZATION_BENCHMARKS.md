# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:24:17 UTC | 8sWfdCTFW8o_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 156.672 | 0.605 | 55.671 | 14.177 | 10.744 | 17.322 | 3.625 |

## 2026-06-24 17:24:17 UTC | 8sWfdCTFW8o_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8sWfdCTFW8o_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `156.672` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.605 |
| save_clips | - |
| sample_frames | 0.948 |
| caption_frames | 41.680 |
| sample_fps | 1.972 |
| detect_object_yolo | 8.527 |
| audio_scan | 15.686 |
| asr_timings | 10.520 |
| ast_timings | 29.456 |
| describe_scenes | 14.177 |
| summarize_scenes | 10.744 |
| synthesize_synopsis | 17.322 |
| make_embedding | 3.625 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 42.634 |
| branch_yolo_total | 10.504 |
| branch_audio_total | 55.671 |
