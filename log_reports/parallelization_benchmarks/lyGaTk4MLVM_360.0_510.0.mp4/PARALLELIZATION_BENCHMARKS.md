# Parallelization Benchmarks

| Timestamp | Video | Mode | Embedding Provider | Embedding Model | Total Wall Sec | Pyscene Sec | Audio and Visual Components | Scene Desc Sec | Narrative Sec | Synopsis Sec | Embedding Sec |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-06-26 17:38:11 UTC | lyGaTk4MLVM_360.0_510.0.mp4 | sequential | gemini | gemini-embedding-001 | 186.748 | 0.724 | 60.007 | 22.253 | 17.341 | 18.754 | 4.476 |

## 2026-06-26 17:38:11 UTC | lyGaTk4MLVM_360.0_510.0.mp4 | sequential

- Video path: `/home/Kairos_model/test/benchmarks/cache/qvh_videos/lyGaTk4MLVM_360.0_510.0.mp4`
- Low memory mode: `True`
- Debug: `False`
- Quiet: `True`
- Embedding provider: `gemini`
- Embedding model: `gemini-embedding-001`
- Total wall time: `186.748` sec

| Step | Wall Time (sec) |
| --- | ---: |
| get_scene_list | 0.724 |
| save_clips | - |
| sample_frames | 1.185 |
| caption_frames | 48.349 |
| sample_fps | 2.266 |
| detect_object_yolo | 9.979 |
| audio_scan | 12.918 |
| asr_timings | 8.301 |
| ast_timings | 38.779 |
| describe_scenes | 22.253 |
| summarize_scenes | 17.341 |
| synthesize_synopsis | 18.754 |
| make_embedding | 4.476 |

| Branch | Wall Time (sec) |
| --- | ---: |
| branch_blip_total | 49.541 |
| branch_yolo_total | 12.251 |
| branch_audio_total | 60.007 |
