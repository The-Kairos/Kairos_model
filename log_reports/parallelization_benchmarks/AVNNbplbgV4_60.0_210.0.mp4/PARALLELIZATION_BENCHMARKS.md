# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-24 19:21:02 UTC | AVNNbplbgV4_60.0_210.0.mp4 | sequential | gemini | gemini-embedding-001 | 119.942 | 0.795 | 49.911 | 8.996 | 7.060 | 15.337 | 2.535 |

## 2026-06-24 19:21:02 UTC | AVNNbplbgV4_60.0_210.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/AVNNbplbgV4_60.0_210.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `119.942` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.795 |
| save_clips | - |
| sample_frames | 0.508 |
| caption_frames | 24.321 |
| sample_fps | 1.957 |
| detect_object_yolo | 7.110 |
| audio_scan | 16.050 |
| asr_timings | 15.625 |
| ast_timings | 18.228 |
| describe_scenes | 8.996 |
| summarize_scenes | 7.060 |
| synthesize_synopsis | 15.337 |
| make_embedding | 2.535 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 24.835 |
| branch_yolo_total | 9.073 |
| branch_audio_total | 49.911 |
