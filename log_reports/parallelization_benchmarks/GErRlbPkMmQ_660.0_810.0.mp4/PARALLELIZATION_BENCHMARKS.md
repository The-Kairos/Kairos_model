# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-25 01:36:56 UTC | GErRlbPkMmQ_660.0_810.0.mp4 | sequential | gemini | gemini-embedding-001 | 161.720 | 0.814 | 54.325 | 15.711 | 11.333 | 21.347 | 3.575 |

## 2026-06-25 01:36:56 UTC | GErRlbPkMmQ_660.0_810.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/GErRlbPkMmQ_660.0_810.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `161.720` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.814 |
| save_clips | - |
| sample_frames | 1.124 |
| caption_frames | 40.653 |
| sample_fps | 2.304 |
| detect_object_yolo | 9.071 |
| audio_scan | 15.017 |
| asr_timings | 9.207 |
| ast_timings | 30.094 |
| describe_scenes | 15.711 |
| summarize_scenes | 11.333 |
| synthesize_synopsis | 21.347 |
| make_embedding | 3.575 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 41.783 |
| branch_yolo_total | 11.381 |
| branch_audio_total | 54.325 |
