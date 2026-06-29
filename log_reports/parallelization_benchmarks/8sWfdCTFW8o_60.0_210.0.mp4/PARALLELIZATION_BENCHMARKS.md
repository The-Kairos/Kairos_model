# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 17:21:40 UTC | 8sWfdCTFW8o_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 175.114 | 0.656 | 58.782 | 12.659 | 12.177 | 26.046 | 3.894 |

## 2026-06-24 17:21:40 UTC | 8sWfdCTFW8o_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/8sWfdCTFW8o_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `175.114` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.656 |
| save_clips | - |
| sample_frames | 1.498 |
| caption_frames | 45.883 |
| sample_fps | 2.297 |
| detect_object_yolo | 9.807 |
| audio_scan | 14.959 |
| asr_timings | 11.313 |
| ast_timings | 32.502 |
| describe_scenes | 12.659 |
| summarize_scenes | 12.177 |
| synthesize_synopsis | 26.046 |
| make_embedding | 3.894 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 47.387 |
| branch_yolo_total | 12.110 |
| branch_audio_total | 58.782 |
